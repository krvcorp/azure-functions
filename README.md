follow this tutorial

https://christianhenrikreich.medium.com/an-easy-setup-of-visual-studio-code-for-running-your-python-azure-functions-on-your-m-arm64-mac-78d4fc6b8386

follow this comment as well

https://github.com/Azure/azure-functions-python-worker/issues/915#issuecomment-965726236

set your python .venv to be

/usr/local/bin/python3.9

commands

arch -x86_64 /usr/local/bin/brew tap azure/functions
arch -x86_64 /usr/local/bin/brew install azure-functions-core-tools@4
