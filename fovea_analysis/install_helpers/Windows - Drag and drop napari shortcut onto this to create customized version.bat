<# :
@echo off
setlocal
set "POWERSHELL_BAT_ARGS=%*"
if defined POWERSHELL_BAT_ARGS set "POWERSHELL_BAT_ARGS=%POWERSHELL_BAT_ARGS:"=\"%"
endlocal & powershell -NoExit -NoLogo -NoProfile -Command "$input | &{ [ScriptBlock]::Create( ( Get-Content \"%~f0\" ) -join [char]10 ).Invoke( @( &{ $args } %POWERSHELL_BAT_ARGS% ) ) }"
goto :EOF
#>

param([string]$linkpath);
Write-Host "powershell argument is " $linkpath

$sh = New-Object -ComObject WScript.Shell
$slnkargs = $sh.CreateShortcut($linkpath).Arguments

Write-Host "link arguments: " $slnkargs

Function GetNapariBatScriptPath($InputString) {
    $regExResult = $InputString | Select-String -CaseSensitive -Pattern '(?<=start '')[A-Z]:.+?(?<=bat)'
    $regExResult.Matches[0].Value
}

$naparibatscriptpath = GetNapariBatScriptPath -InputString $slnkargs

Write-Host "napari bat script path: " $naparibatscriptpath


Function GetFileName()
{
 [System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null
 $OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog
 $OpenFileDialog.filter = "Locate main_app (*.py)| *.py"
 $OpenFileDialog.ShowDialog() | Out-Null
 $OpenFileDialog.filename
}
$custompyscript = GetFileName
$custompydir = ([System.IO.DirectoryInfo]$custompyscript).Parent.FullName
$custompyreqs = Join-Path -Path $custompydir -ChildPath "install_helpers/requirements_tighter.txt"
$custompyreqinstallbat = Join-Path -Path $custompydir -ChildPath "install_custom_requirements.bat"
$custompylaunchbat = Join-Path -Path $custompydir -ChildPath "LAUNCH_CUSTOMIZED_NAPARI.bat"

Write-Host "given python script: " $custompyscript
Write-Host "custompydir: " $custompydir
Write-Host "custompyreqs: " $custompyreqs

Copy-Item -Path $naparibatscriptpath -Destination $custompyreqinstallbat
$content = [System.IO.File]::ReadAllText($custompyreqinstallbat).Replace("python.exe -m napari", "python.exe -m pip install --only-binary :all: --prefer-binary -r $custompyreqs" )
[System.IO.File]::WriteAllText($custompyreqinstallbat, $content)

Copy-Item -Path $naparibatscriptpath -Destination $custompylaunchbat
$content = [System.IO.File]::ReadAllText($custompylaunchbat).Replace("python.exe -m napari", "python.exe $custompyscript" )
[System.IO.File]::WriteAllText($custompylaunchbat, $content)

cmd.exe /c $custompyreqinstallbat

