#!/usr/bin/env bash

HERE=$(pwd)

source $HERE/.venv/bin/activate

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_help() {
    echo ""
    echo -e "${YELLOW}Usage:${NC} $0 <model_number_or_name>"
    echo ""
    echo -e "${CYAN}Available models:${NC}"
    echo -e "  ${GREEN}0)${NC} Hinata (Swift-1.5-Qwen3.8-Flash-Next-W4A16-Attn8-FP8PLE)"
    echo -e "  ${GREEN}1)${NC} Scarlet (Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin)"
    echo -e "  ${GREEN}1)${NC} Yuuki (Swift-1.5-Qwen3.8-27B-INT8-W8A16-AutoRound-MTP)"
    echo -e "  ${GREEN}1)${NC} Emma (Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)"
    echo ""
    exit 1
}

if [ -z "$1" ]; then
    show_help
fi

case "$1" in
    0|Hinata) MODEL="Hinata - Swift 1.5 Qwen 3.8 Flash Next" ;;
    1|Scarlet) MODEL="Scarlet - Swift 1.5 Qwen 3.8 Flash Next" ;;
    2|Yuuki) MODEL="Yuuki - Swift 1.5 Qwen 3.8 27B" ;;
    3|Emma) MODEL="Emma - Qwen 3.6 35B-A3B" ;;
    *)
        echo ""
        echo -e "${RED}Error:${NC} Invalid model '$1'"
        show_help
        ;;
esac

echo -e "${GREEN}Running benchmark for ${CYAN}${MODEL}${NC}..."
llmbench run --with-l6 --with-l7 --runs 3 --model "$MODEL"
echo ""
