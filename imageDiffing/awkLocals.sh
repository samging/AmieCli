#!/bin/bash

filterLocal() {
	BASEDIR=$(pwd)	
	localFile="$BASEDIR/localAmieLogs.txt"
	if [ -f $localFile ]; then
		echo "appending to existing file"	
	else
		$(touch $localFile)
	fi
	content=$(awk -F '/\.local' '/\/\.local/ {print}' "$BASEDIR/v2_v3_changes.txt" > $localFile)
}

filterLocal
