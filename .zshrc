alias arm="env /usr/bin/arch -arm64 /bin/zsh --login"
alias intel="env /usr/bin/arch -x86_64 /bin/zsh --login"

if [ $(arch) = "i386" ]; then
    alias brew='/usr/local/bin/brew'
else
    alias brew='/opt/homebrew/bin/brew'
fi