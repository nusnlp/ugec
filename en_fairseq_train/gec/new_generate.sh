fairseq-interactive data-bin \
    --path $1 \
    --beam 5 \
    --nbest 5 \
    --bpe gpt2 \
    --source-lang src \
    --target-lang tgt \
    --log-format simple \
    --remove-bpe \
    --max-tokens 50000 \
    --buffer-size 50000 \
    --batch-size 32 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 < $2 > $2.nbest.tok

cat $2.nbest.tok | grep "^S"  | cut -f2 > $2.source.tok
cat $2.nbest.tok | grep "^D"  | python3 -c "import sys; x = sys.stdin.readlines(); x = [t for i, t in enumerate(x) if i % 5 == 0]; sys.stdout.writelines(x)"  | cut -f3 > $2.best.tok
cat $2.nbest.tok | grep "^D"  | python3 -c "import sys; x = sys.stdin.readlines(); x = [t for i, t in enumerate(x) if i % 5 == 0]; sys.stdout.writelines(x)"  | cut -f2 > $2.best.sentscore
cat $2.nbest.tok | grep "^P"  | python3 -c "import sys; x = sys.stdin.readlines(); x = [t for i, t in enumerate(x) if i % 5 == 0]; sys.stdout.writelines(x)"  | cut -f2 > $2.best.tokscore
