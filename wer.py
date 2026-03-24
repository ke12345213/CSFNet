def wer(reference: str, hypothesis: str) -> float:

    r = reference.strip().split()
    h = hypothesis.strip().split()
    n = len(r)

    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitute = dp[i - 1][j - 1] + 1
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                dp[i][j] = min(substitute, insert, delete)

    return dp[len(r)][len(h)] / n if n > 0 else float('inf')

ref = "I have a dream"
hyp = "I had a cream"
print(f"WER: {wer(ref, hyp):.2%}")
