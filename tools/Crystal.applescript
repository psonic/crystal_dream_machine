-- Crystal.app: apre il pannello di controllo (o lo porta davanti se è già aperto).
-- Si ricompila con:  tools/make_app.sh   (che sostituisce __PROJECT__ con la cartella del progetto)
set projDir to "__PROJECT__"
set py to projDir & "/.venv/bin/python"
try
	do shell script "test -x " & quoted form of py
on error
	display dialog "Non trovo il Python di Crystal in:" & return & py & return & return & "Il disco del progetto è collegato? Se hai spostato la cartella, rilancia tools/make_app.sh." buttons {"OK"} default button 1 with icon caution with title "Crystal"
	return
end try
do shell script "cd " & quoted form of projDir & " && mkdir -p .cache/run && (nohup ./.venv/bin/python -m crystal panel > .cache/run/panel.log 2>&1 &)"
