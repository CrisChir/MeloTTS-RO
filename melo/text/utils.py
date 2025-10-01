def distribute_phone(phone_len, word_len):
    """
    Distributes the phoneme count across the word's sub-tokens.
    For example, if a word has 5 phonemes and 2 sub-tokens,
    it might distribute them as [3, 2].
    """
    phones_per_word = phone_len // word_len
    remaining_phones = phone_len % word_len
    phone_distribution = []
    for i in range(word_len):
        phones = phones_per_word
        if i < remaining_phones:
            phones += 1
        phone_distribution.append(phones)
    return phone_distribution
