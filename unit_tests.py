import utilities as pu


def test_countGrams():
    reviews_raw = [{"text": "Check this out."},
                   {"text": "Check this out, please."}]
    result = pu.countTotalGrams(reviews_raw, 2, 3)
    print(result)


if __name__ == "__main__":
    test_countGrams()
