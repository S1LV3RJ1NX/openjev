"""Banking77 option descriptions, copied verbatim from the measurement run.

GENERATED, vendored verbatim from
our Banking77 option-description ablation. Do not hand-edit:
the point of copying rather than rewriting is that this exact text is the arm
that scored 0.8767 against a 0.8200 bare baseline on Banking77 (+5.1 points
pooled over 410 items, 95% CI [+2.2, +8.3], p = 0.0011). Rewording it would
silently invalidate the comparison.

Provenance: written by reading the Banking77 TRAIN split only. No test item and
no test label was inspected while writing these.
"""

INSTRUCTIONS = (
    'This is a single message from a customer of a digital-only bank, sent to customer '
    'support. Which one of the following intents best describes what the customer is '
    'asking about?'
)

BANKING77: dict[str, str] = {
    'Refund_not_showing_up':
        'A refund was already requested or promised and has still not appeared on the '
        'statement.',
    'activate_my_card':
        'The customer has the card in hand and wants to activate it or start using it.',
    'age_limit':
        'The minimum age for an account, or opening an account for a child.',
    'apple_pay_or_google_pay':
        'Using Apple Pay or Google Pay, including setting them up and topping up with them.',
    'atm_support':
        'Which ATMs can be used, or where the customer can withdraw money.',
    'automatic_top_up':
        'Setting up, finding or limiting automatic or recurring top-ups.',
    'balance_not_updated_after_bank_transfer':
        'The customer transferred money into their own account and their own balance has not '
        'gone up yet.',
    'balance_not_updated_after_cheque_or_cash_deposit':
        'A cash or cheque deposit has not yet appeared in the balance.',
    'beneficiary_not_allowed':
        'A transfer was blocked because the recipient or beneficiary is not permitted.',
    'cancel_transfer':
        'The customer wants to stop, cancel or undo a transfer they set up, often because it '
        'was sent to the wrong account.',
    'card_about_to_expire':
        'The card is expiring or has expired and the customer asks about its replacement.',
    'card_acceptance':
        'Which shops or businesses accept the card.',
    'card_arrival':
        'The customer already ordered a card and is chasing that specific one: it has not '
        'turned up, they ask where it is, how long it has been, or whether it can be tracked.',
    'card_delivery_estimate':
        'A general question about how long card delivery takes, or when a card will be '
        'received, phrased as a duration or timeframe, including delivery times to a particular '
        'country. The customer is not tracking a specific overdue card.',
    'card_linking':
        'Linking, adding or re-activating a card inside the app, typically a card the customer '
        'already holds, including re-enabling one they had frozen and then found again.',
    'card_not_working':
        'A physical card is being rejected generally or repeatedly, described as something '
        'wrong with the card itself rather than with one particular payment.',
    'card_payment_fee_charged':
        'A fee charged on top of a card payment or purchase.',
    'card_payment_not_recognised':
        'A card payment on the statement that the customer says they did not make.',
    'card_payment_wrong_exchange_rate':
        'The exchange rate applied to a card payment or purchase was wrong.',
    'card_swallowed':
        'An ATM has retained or eaten the card and the customer wants it back.',
    'cash_withdrawal_charge':
        'A fee charged for withdrawing cash at an ATM.',
    'cash_withdrawal_not_recognised':
        'A cash withdrawal on the statement that the customer did not make.',
    'change_pin':
        'The customer already has a PIN and wants to change it, or asks where and how a PIN can '
        'be changed. Not for a PIN that has never arrived.',
    'compromised_card':
        'The customer suspects fraud, a security breach, or that somebody else is using their '
        'card, while the card itself is still in their possession.',
    'contactless_not_working':
        'Contactless payment specifically is failing, or the customer asks how to set '
        'contactless up.',
    'country_support':
        'Which countries the service is available in, or whether the customer can hold an '
        'account or card while living in or moving to a particular country.',
    'declined_card_payment':
        'One particular card payment was declined or refused.',
    'declined_cash_withdrawal':
        'A cash or ATM withdrawal was declined or refused.',
    'declined_transfer':
        'A transfer was explicitly declined or refused.',
    'direct_debit_payment_not_recognised':
        'An unrecognised payment where the message says direct debit, or describes a seller or '
        'merchant taking money that was never approved.',
    'disposable_card_limits':
        'Limits on disposable or virtual cards: how many can be created, or how many '
        'transactions each one allows.',
    'edit_personal_details':
        'Changing personal details held on the account, such as address or name.',
    'exchange_charge':
        'The fee for exchanging currency.',
    'exchange_rate':
        'How the exchange rates are set, or what rate is applied and when.',
    'exchange_via_app':
        'Whether the app itself can convert or exchange one currency into another.',
    'extra_charge_on_statement':
        'A small extra charge of roughly one pound, dollar or euro on the statement. Use this '
        'only for that specific small extra amount, not for fees in general.',
    'failed_transfer':
        'A transfer the customer attempted did not go through or errored.',
    'fiat_currency_support':
        'Which currencies are supported, accepted or can be held.',
    'get_disposable_virtual_card':
        'Obtaining disposable (single-use) virtual cards, or what they are for.',
    'get_physical_card':
        'The customer is asking about the PIN for their card: they have not received a PIN, '
        'cannot find it, or want to know where it is or when it will arrive. Use this for any '
        "'where is my PIN' or 'I do not have a PIN yet' message. Despite the name, this is not "
        'about requesting a card.',
    'getting_spare_card':
        'The customer wants an extra or additional card on top of one they already have, '
        'including a second card for a family member, and what extra cards cost.',
    'getting_virtual_card':
        'Obtaining or finding a virtual card, where the message does not say disposable.',
    'lost_or_stolen_card':
        'The physical card itself has been lost or stolen and the customer is reporting it.',
    'lost_or_stolen_phone':
        'The phone carrying the app was lost or stolen.',
    'order_physical_card':
        'The customer does not have a physical card yet and wants one: how to get or request '
        'one, whether one can be sent, where cards can be delivered to, or what a card costs. '
        'Use this for the initial request for a card, before any card has been ordered.',
    'passcode_forgotten':
        'The app passcode or password is forgotten or rejected, blocking access to the app.',
    'pending_card_payment':
        'A card payment shows as pending.',
    'pending_cash_withdrawal':
        'A cash withdrawal shows as pending or has not appeared yet.',
    'pending_top_up':
        'A top-up shows as pending, or money paid in is not available yet.',
    'pending_transfer':
        'A transfer shows as pending.',
    'pin_blocked':
        'The PIN is blocked or forgotten, or was entered wrong too many times, and needs '
        'resetting or unblocking.',
    'receiving_money':
        'Money arriving from somebody else: whether the customer can be paid by a friend, a '
        'buyer or an employer, receive a direct deposit, or be paid in another currency. Not '
        'the customer moving their own money in.',
    'request_refund':
        'The customer wants their money back for something they bought, or asks about the '
        'refund or return policy.',
    'reverted_card_payment?':
        "A card payment was reversed: the money came back into the customer's account, or the "
        'seller says they were never paid even though the amount left the account.',
    'supported_cards_and_currencies':
        'Which card brands and which currencies can be used to add money to the account.',
    'terminate_account':
        'Closing, deleting or ending the account.',
    'top_up_by_bank_transfer_charge':
        'What it costs to add money or receive money by bank transfer, including SEPA and SWIFT '
        'transfers, and whether those transfer types are accepted.',
    'top_up_by_card_charge':
        'What a card top-up costs: the fee for adding money with a debit or credit card.',
    'top_up_by_cash_or_cheque':
        'Adding money with physical cash or a cheque, including where to deposit it and how '
        'long a cheque takes to clear.',
    'top_up_failed':
        'A top-up attempt errored, was denied, or did not go through.',
    'top_up_limits':
        'The maximum amount or frequency allowed for top-ups, including changing that limit.',
    'top_up_reverted':
        'A top-up was reverted, cancelled or rejected after having been made.',
    'topping_up_by_card':
        'How to add money to the account with a card, including whether somebody else can top '
        'the account up, and card top-ups that have not shown up in the wallet.',
    'transaction_charged_twice':
        'The same purchase was charged twice: a duplicate of one transaction.',
    'transfer_fee_charged':
        'A fee charged on a transfer the customer made.',
    'transfer_into_account':
        'The customer moving their own money into this account from another bank account by '
        'bank transfer, including how the bank transfer process works.',
    'transfer_not_received_by_recipient':
        'The customer sent money to somebody else and that recipient has not received it.',
    'transfer_timing':
        'A general question about how long a transfer takes, often naming a country or region, '
        'with no complaint that one specific transfer has gone missing.',
    'unable_to_verify_identity':
        'Identity verification was attempted and failed, or the customer cannot complete it.',
    'verify_my_identity':
        'The customer asks how to verify their identity or what documents are accepted.',
    'verify_source_of_funds':
        "Verifying or explaining where the customer's money has come from.",
    'verify_top_up':
        'The verification code needed to confirm the card used for a top-up.',
    'virtual_card_not_working':
        'The same problem, but the message explicitly says the card is virtual or disposable.',
    'visa_or_mastercard':
        'Which card scheme or brand the customer will be issued or may choose: Visa, '
        'Mastercard, or whether the brand matters.',
    'why_verify_identity':
        'The customer asks why the bank needs so much personal information, or what the '
        'identity check is for. A question about the reason, not about carrying it out.',
    'wrong_amount_of_cash_received':
        'The ATM dispensed less cash than the customer asked for.',
    'wrong_exchange_rate_for_cash_withdrawal':
        'The exchange rate applied to a cash or ATM withdrawal was wrong.',
}
