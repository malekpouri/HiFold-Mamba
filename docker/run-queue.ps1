# HiFold-Mamba Stage 2 queue for Windows + Docker Desktop: runs one training container at a time.
#
#   powershell -ExecutionPolicy Bypass -File .\run-queue.ps1           run the queue
#   powershell -ExecutionPolicy Bypass -File .\run-queue.ps1 -DryRun   only list what would start or be skipped
#
# Safe to run again at any time:
#   - a run that already finished all its steps is skipped;
#   - a run whose inputs are not ready yet is skipped with a reason (the Base fork file, a Base checkpoint,
#     the H3 data); run the script again once they exist;
#   - an unfinished run starts again from step 0 (there is no resume).
# Closing the window stops the queue only; the current run keeps going.
# Start it only AFTER the OSF registration.

param([switch]$DryRun)

$WORK = "D:\hifold\work"
$IMG  = "hifold-mamba:IMAGE_TAG"
$TZ   = "Asia/Tehran"

# ---- the planned runs, in order ---------------------------------------------------------------------
# name, config, seed, expected steps, needs, overrides
#   needs: ""        nothing
#          "fork"    hifold_paper seed 0 has saved fork_step52560.pt   (lambda pilot)
#          "base"    the seed-matched hifold_paper run has finished     (H3 pretrain arm)
#          "h3data"  /data/cache/h3 is in the Docker volume              (all H3 runs)
#          "test5"   /data/cache/akita5/test is in the Docker volume     (chromosome-held-out runs)
$runs = New-Object System.Collections.ArrayList
function Add-Run($name, $config, $seed, $steps, $needs, $over) {
  [void]$runs.Add([pscustomobject]@{ Name=$name; Config=$config; Seed=$seed; Steps=$steps; Needs=$needs; Over=$over })
}

# 1. Base seed 0 first: it saves the fork state for the lambda pilot
Add-Run "hifold_paper" "hifold_paper" 0 70080 "" @()
# 2. lambda pilot (manuscript sec. 2.6): fork Base seed 0 at step 52,560, Struct phase for each (lambda_ins, lambda_bnd)
$k = 0
foreach ($li in "0.1", "0.3", "1.0") { foreach ($lb in "0.1", "0.3", "1.0") {
  $k++
  Add-Run "lambda_pilot_$k" "hifold_struct" 0 70080 "fork" @("struct.ins=$li", "struct.bnd=$lb")
} }
# 3. Base seeds 1 and 2
Add-Run "hifold_paper" "hifold_paper" 1 70080 "" @()
Add-Run "hifold_paper" "hifold_paper" 2 70080 "" @()
# 4. baselines and ablations
foreach ($c in "akita_repro", "baseline_attention", "baseline_conv", "ablation_norc", "ablation_nomixer") {
  $st = if ($c -eq "akita_repro") { 140160 } else { 70080 }
  foreach ($s in 0, 1, 2) { Add-Run $c $c $s $st "" @() }
}
# 5. H3 (manuscript sec. 2.11): 4 lines x 4 regimes x 2 arms x 3 seeds
foreach ($line in "HAP1", "HeLa", "HepG2", "K562") { foreach ($r in "10", "25", "50", "full") {
  foreach ($s in 0, 1, 2) {
    Add-Run "h3_${line}_${r}_pre"     "h3_finetune" $s 3504 "base,h3data" @("data.lines=[$line]", "data.subset_regime=$r")
    Add-Run "h3_${line}_${r}_scratch" "h3_finetune" $s 3504 "h3data"      @("data.lines=[$line]", "data.subset_regime=$r")
  }
} }
# 6. chromosome-held-out robustness split (manuscript sec. 2.13), one seed
foreach ($c in "hifold_paper", "akita_repro", "baseline_attention", "baseline_conv") {
  $st = if ($c -eq "akita_repro") { 140160 } else { 70080 }
  Add-Run "robust_$c" $c 0 $st "test5" @("data.split_scheme=chrom_holdout", "save_at=[]")
}
# Struct (3 seeds), the single-term ablations and w = 20 / 80 are added after the lambda selection.

# ---- helpers -----------------------------------------------------------------------------------------
function Get-Status($dir) {
  try { Get-Content "$dir\status.json" -Raw -ErrorAction Stop | ConvertFrom-Json } catch { $null }
}
function Find-Finished($name, $seed, $steps) {
  Get-ChildItem "$WORK\runs" -Directory -Filter "$name-*-s$seed" -ErrorAction SilentlyContinue |
    Where-Object { $s = Get-Status $_.FullName; $s -and $s.state -eq "finished" -and $s.steps -eq $steps } |
    Sort-Object LastWriteTime | Select-Object -Last 1
}

# ---- checks before starting -------------------------------------------------------------------------
$busy = docker ps --filter "ancestor=$IMG" --format "{{.Names}}" | Where-Object { $_ -ne "hifold-dash" }
if ($busy) { Write-Host "A training container is already running: $busy. Queue not started."; exit 1 }
$data = docker run --rm -v hifold-cache:/data/cache:ro $IMG sh -c "ls -d /data/cache/h3/train /data/cache/h3/valid /data/cache/akita5/test 2>/dev/null"
$hasH3 = [bool]($data | Select-String "h3/train") -and [bool]($data | Select-String "h3/valid")
$hasTest5 = [bool]($data | Select-String "akita5/test")
Write-Host "H3 data: $hasH3   akita5/test: $hasTest5   planned runs: $($runs.Count)"
New-Item -ItemType Directory -Force "$WORK\runs" | Out-Null

# ---- the queue ---------------------------------------------------------------------------------------
$started = 0; $skipped = @()
foreach ($r in $runs) {
  $label = "$($r.Name) seed $($r.Seed)"
  if (Find-Finished $r.Name $r.Seed $r.Steps) { continue }

  $extra = @()
  $why = $null
  foreach ($n in ($r.Needs -split "," | Where-Object { $_ })) {
    switch ($n) {
      "h3data" { if (-not $hasH3) { $why = "H3 data not in the Docker volume" } }
      "test5"  { if (-not $hasTest5) { $why = "akita5/test not in the Docker volume" } }
      "fork" {
        $f = Get-ChildItem "$WORK\runs" -Recurse -Filter "fork_step52560.pt" -ErrorAction SilentlyContinue |
             Where-Object { $_.Directory.Name -like "hifold_paper-*-s0" } | Select-Object -Last 1
        if (-not $f) { $why = "Base seed 0 has not reached step 52,560 yet" }
        else { $extra += "resume_from=/work/runs/$($f.Directory.Name)/fork_step52560.pt" }
      }
      "base" {
        $b = Find-Finished "hifold_paper" $r.Seed 70080
        if (-not $b) { $why = "Base seed $($r.Seed) has not finished yet" }
        else { $extra += "init_from=/work/runs/$($b.Name)/best.pt" }
      }
    }
  }
  if ($why) { $skipped += [pscustomobject]@{ Run = $label; Why = $why }; continue }

  if ($DryRun) {
    Write-Host ("would start  {0,-26} {1} {2}" -f $label, $r.Config, (($r.Over + $extra) -join " "))
    $started++; continue
  }
  # without a GPU, training would silently run on the CPU: check before every run
  $check = docker run --rm --gpus all $IMG check 2>&1 | Out-String
  if ($check -notmatch "bf16 native True") { Write-Host $check; Write-Host "GPU check failed. Queue stopped."; exit 1 }

  $cname = ("{0}-s{1}" -f $r.Name, $r.Seed)
  docker rm $cname 2>$null | Out-Null
  $cmd = @("run", "-d", "--name", $cname, "--gpus", "all", "--shm-size=16g", "-e", "TZ=$TZ", "-e", "HOME=/tmp",
           "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
           "-v", "hifold-cache:/data/cache:ro", "-v", "${WORK}:/work", $IMG,
           "train", $r.Config, "seed=$($r.Seed)", "name=$($r.Name)") + $r.Over + $extra
  Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  start $label"
  docker @cmd | Out-Null
  $code = docker wait $cname
  Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  end   $label (exit code $code)"
  $started++
}
Write-Host ""
if ($DryRun) { Write-Host "Dry run: $started run(s) would start now (later dependencies are not simulated)." }
else { Write-Host "Queue pass finished: $started run(s) started this time." }
if ($skipped) {
  Write-Host "Not ready (run the script again later):"
  $skipped | Group-Object Why | ForEach-Object {
    $first = $_.Group[0].Run
    Write-Host ("  {0,3} run(s): {1}   (first: {2})" -f $_.Count, $_.Name, $first)
  }
}
