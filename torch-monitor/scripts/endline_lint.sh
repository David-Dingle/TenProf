#!/bin/bash

exit_code=0

for file in $(find . -type f ! -path "./.git/*" ! -name "$(basename -- "$0")"); do
    if ! [ -z "$(tail -c 1 "$file")" ]; then
        echo "No newline at end of $file"
        exit_code=1
    fi
done

exit $exit_code
