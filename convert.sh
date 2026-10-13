#!/usr/bin/env bash

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo ""
    echo -e "${RED}Error:${NC} No input provided."
    echo -e "Usage: $0 <input.md> [output.html] OR pass flags directly"
    echo ""
    exit 1
fi

echo ""
echo -e "${CYAN}Converting Markdown to HTML...${NC}"

if [ $# -eq 1 ] && [[ "$1" != -* ]]; then
    output="${1%.*}.html"
    bunx markdown-to-html-cli -s "$1" -o "$output"
elif [ $# -eq 2 ] && [[ "$1" != -* ]]; then
    bunx markdown-to-html-cli -s "$1" -o "$2"
else
    bunx markdown-to-html-cli "$@"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Done!${NC}"
else
    echo -e "${RED}Conversion failed.${NC}"
fi

echo ""
