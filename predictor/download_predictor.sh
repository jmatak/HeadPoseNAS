wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz" \
    -O 20180408-102900.zip && rm -rf /tmp/cookies.txt
unzip 20180408-102900.zip
mv 20180408-102900/20180408-102900.pb predictor.pb
rm -r 20180408-102900*