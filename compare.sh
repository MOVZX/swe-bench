#!/usr/bin/env bash

source /data/swe-bench/.venv/bin/activate

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

FILES=(results/*_results.json)

if [ ! -e "${FILES[0]}" ]; then
    echo ""
    echo -e "${RED}Error:${NC} No result files found matching results/*_results.json"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}Comparing results...${NC}"
llmbench compare "${FILES[@]}"
echo ""
