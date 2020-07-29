#!/bin/bash
set -u -e -o pipefail

declare REPO=$1
declare DIR=$2

if [[ ! -d $DIR ]]; then
    git clone $REPO $DIR

    git config --global user.email "gogs@gogs.com"
fi
