#!/usr/bin/sh

DIR=/local/home/lo272082/drift-velocity-measurements/output
while inotifywait -r -e modify,moved_to,create,delete $DIR; do
rsync -avzP $DIR/* sultan:/storage/iaxo/picosecond-drift-velocity/data
done
