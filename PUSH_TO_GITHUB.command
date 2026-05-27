#!/bin/bash
cd "$(dirname "$0")"
echo "==========================================="
echo "Pushing to https://github.com/ChaitanyaK77/LLM-Post-Training-"
echo "==========================================="
echo ""
echo "When prompted:"
echo "  Username: ChaitanyaK77"
echo "  Password: paste a GitHub Personal Access Token"
echo "  (get one at https://github.com/settings/tokens)"
echo ""
git push -u origin main --force
echo ""
echo "Done. Press any key to close..."
read -n 1
