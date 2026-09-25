from src.normalize import (
    basic_clean,
    norm_name,
    core_name,
    norm_addr,
    postal_codes,
    numbers,
)


def test_accents_normalize_identically():
    assert norm_name("Société Générale") == norm_name("Societe Generale")
    assert norm_name("Café") == norm_name("Cafe")


def test_legal_suffixes_are_removed_from_core_name():
    assert core_name("ABC Pvt. Ltd.") == "abc"
    assert core_name("ABC Private Limited") == "abc"
    assert core_name("ABC LLC") == "abc"


def test_french_legal_suffixes_are_removed_from_core_name():
    assert core_name("ABC SARL") == "abc"
    assert core_name("ABC SAS") == "abc"
    assert core_name("ABC SASU") == "abc"
    assert core_name("ABC EURL") == "abc"


def test_all_legal_tokens_fall_back_to_full_normalized_name():
    assert core_name("The Company Ltd") == "the company ltd"


def test_punctuation_and_abbreviations():
    assert norm_name("A&B") == "a and b"
    assert norm_name("p.v.t.") == "pvt"


def test_st_has_context_specific_meaning():
    assert norm_name("St Martin") == "saint martin"
    assert norm_addr("10 St Martin Road") == "10 street martin road"


def test_postal_codes_accept_5_and_6_digits_only():
    assert postal_codes("Paris 75001") == {"75001"}
    assert postal_codes("Mumbai 400001") == {"400001"}
    assert postal_codes("1234") == set()
    assert postal_codes("1234567") == set()


def test_numbers_extract_all_digit_runs():
    assert numbers("12 Rue de Paris 75001") == {"12", "75001"}


def test_empty_and_missing_strings_do_not_raise():
    assert basic_clean("") == ""
    assert basic_clean("   ") == ""
    assert basic_clean("!!!") == ""
    assert norm_name("") == ""
    assert core_name("") == ""
    assert norm_addr("") == ""
    assert postal_codes("") == set()
    assert numbers("") == set()


def test_address_single_letters_are_not_expanded_as_directions():
    assert norm_addr("80 Quai des Queyries, Bat E, Apt 15") == (
        "80 quai des queyries bat e apartment 15"
    )
    assert norm_addr("25 Rue dAjaccio, Bat N, Apt 11") == (
        "25 rue dajaccio bat n apartment 11"
    )