"""Question text for the held-out suite: instructions and per-option criteria.

Every string in this file was written from one of exactly two sources:

* the dataset's own **documentation** (dataset card, paper, annotation
  guidelines), or
* its **train split** (for the intent sets, via the per-label dumps produced by
  `scripts/build_heldout.py --dump-train`).

**No test split was read while writing any of this.** That is not a stylistic
note. `docs/04-open-questions.md` records that discriminative option text is
worth about +5.1 accuracy points on Banking77, so a description contaminated by
a test label would inflate the held-out number by a margin larger than most of
the effects this project cares about.

Where two labels are near-twins the description says what *separates* them
rather than restating the name, because mechanical restatement was measured to
be worth exactly nothing (p = 0.75 at n = 300).

Label keys are the source dataset's own strings, verbatim, including the ones
that are misleading (`get_physical_card` is about PINs) or oddly cased
(`Refund_not_showing_up`). Renaming them would change the task: see
`docs/architecture.md` on label surface form carrying semantic weight. The
single exception is CLINC's `oos`, which the brief asks to be surfaced as the
`none_of_the_above` escape option; it is called out in the README.
"""

from __future__ import annotations

from heldout_criteria_banking77 import BANKING77, INSTRUCTIONS as BANKING77_INSTRUCTIONS

__all__ = [
    "BANKING77",
    "BANKING77_INSTRUCTIONS",
    "CLINC_INSTRUCTIONS",
    "CLINC",
    "CLINC_NONE_OF_THE_ABOVE",
    "MASSIVE_INSTRUCTIONS",
    "MASSIVE",
    "AG_NEWS_INSTRUCTIONS",
    "AG_NEWS",
    "SST5_INSTRUCTIONS",
    "SST5",
    "CIVIL_INSTRUCTIONS",
    "CIVIL",
    "HELPSTEER_INSTRUCTIONS",
    "HELPSTEER",
]


# ==========================================================================
# CLINC150 / clinc_oos  --  choice, 150 in-scope intents + an escape option
# ==========================================================================
#
# Source: `clinc/clinc_oos` config `plus`, train split. The 150 intents span
# ten domains (banking, credit cards, kitchen and dining, home, auto, travel,
# utility, work, small talk, meta). The hard pairs here are the ones that
# differ only by read-versus-write or by how-versus-when, so those
# descriptions lead with that distinction.

CLINC_INSTRUCTIONS = (
    'This is a single utterance spoken to a general-purpose voice assistant. Which one of '
    'the following intents is the user expressing? Choose "none_of_the_above" if the '
    'utterance is outside everything this assistant handles, which happens often: many '
    'users ask for things the assistant was never built to do.'
)

CLINC_NONE_OF_THE_ABOVE = (
    'The utterance is out of scope. It is a reasonable thing for a person to say, and it '
    'may even be phrased like a normal assistant request, but it is not one of the other '
    'intents on this list. Choose this rather than forcing the closest-sounding intent: '
    'the request is for a capability the assistant does not have, a topic it does not '
    'cover, or information it cannot hold.'
)

CLINC: dict[str, str] = {
    # -- banking -----------------------------------------------------------
    'account_blocked':
        'The account is already frozen, barred or on hold and the user wants to know why '
        'they cannot get at it. The block exists and they did not ask for it.',
    'balance':
        'How much money is currently in a bank account. A plain balance lookup, not a bill '
        'and not reward points.',
    'bill_balance':
        'How much is owed on a specific bill from a named provider such as a phone, cable '
        'or electricity company. The amount owed to a third party, not money held at the bank.',
    'bill_due':
        'The date a specific bill, loan or insurance payment is next due.',
    'freeze_account':
        'The user is asking the bank to put a stop, hold or freeze on their account. A '
        'request they are making now, not an existing block they are complaining about.',
    'interest_rate':
        'The interest rate paid or charged on a bank account such as checking or savings. '
        'Not a credit card rate, which is apr.',
    'min_payment':
        'The smallest amount the user is allowed to pay on a bill this cycle.',
    'order_checks':
        'Ordering more paper checks or checkbooks to be mailed out.',
    'pay_bill':
        'An instruction to actually make a payment now, rather than to look up an amount '
        'or a date.',
    'pin_change':
        'Changing, resetting or obtaining a new PIN number.',
    'report_fraud':
        'Reporting fraudulent or unauthorised activity on an account or card. The card is '
        'not necessarily missing; the complaint is about charges the user did not make.',
    'routing':
        'The bank routing number for an account.',
    'spending_history':
        'How much the user spent over some past period, or at some named merchant. An '
        'aggregate over time, not a list of individual purchases.',
    'transactions':
        'A list or lookup of individual recent purchases on the account, such as the last '
        'five, or the most recent one.',
    'transfer':
        'Moving money between accounts, whether the user\'s own or someone else\'s.',

    # -- credit cards ------------------------------------------------------
    'application_status':
        'Whether a credit card or account application the user already submitted has been '
        'approved, processed or gone through yet.',
    'apr':
        'The annual percentage rate or interest rate charged on a credit card.',
    'card_declined':
        'Why a card was refused at a specific purchase that already happened.',
    'credit_limit':
        'What the current spending limit on a card is. A lookup of an existing number.',
    'credit_limit_change':
        'A request to raise, lower or otherwise alter the credit limit. A change, not a lookup.',
    'credit_score':
        'What the user\'s current credit score or rating is. A lookup of an existing number.',
    'improve_credit_score':
        'How to build, protect or repair credit, or what actions would harm it. Advice '
        'about the future, not a lookup of the current score.',
    'damaged_card':
        'The physical card is broken, scratched, shredded or chewed up, and needs replacing. '
        'The card is present but unusable.',
    'expiration_date':
        'When a specific card expires.',
    'international_fees':
        'What it costs to use the card abroad, including foreign ATM and transaction fees.',
    'new_card':
        'How to obtain or apply for a card the user does not have yet. A first card, not a '
        'replacement for one they lost.',
    'redeem_rewards':
        'Cashing in, spending or using credit card reward points.',
    'replacement_card_duration':
        'How long a replacement card will take to arrive. A question about elapsed time.',
    'report_lost_card':
        'Reporting a card as lost or stolen so it can be cancelled.',
    'rewards_balance':
        'How many reward points the user has accumulated. A lookup of a points total.',

    # -- kitchen and dining ------------------------------------------------
    'accept_reservations':
        'Whether a named restaurant takes reservations at all. A question about the '
        'restaurant\'s policy, not a booking.',
    'calories':
        'How many calories are in a food or dish.',
    'cancel_reservation':
        'Cancelling or getting rid of a restaurant reservation that already exists.',
    'confirm_reservation':
        'Confirming or verifying a reservation that already exists, usually naming a time '
        'and a name.',
    'cook_time':
        'How long something needs to cook, bake or simmer. A duration.',
    'food_last':
        'How long food stays good, or when it will spoil or expire.',
    'ingredient_substitution':
        'Whether one ingredient can be used in place of another.',
    'ingredients_list':
        'What ingredients a dish contains or what is needed to make it. The list of items, '
        'not the method.',
    'meal_suggestion':
        'Asking what dish or cuisine to eat or make, with no restaurant involved.',
    'nutrition_info':
        'The nutritional content or healthiness of a food, beyond just its calorie count.',
    'recipe':
        'A request for a recipe or for how to cook a named dish. The method, not the '
        'ingredient list alone.',
    'restaurant_reservation':
        'Booking a new table at a restaurant.',
    'restaurant_reviews':
        'What reviews or ratings say about a named restaurant.',
    'restaurant_suggestion':
        'Asking which restaurant or place to eat at, usually in a named area.',
    'how_busy':
        'How crowded a place is or how long the wait will be, usually at a named time.',

    # -- home --------------------------------------------------------------
    'calendar':
        'Reading the calendar: what is on it, or whether some event is on it.',
    'calendar_update':
        'Changing the calendar by adding or removing an event.',
    'date':
        'What today\'s date is, or what the date will be some number of days out.',
    'find_phone':
        'The user has misplaced their phone and wants help locating it.',
    'how_old_are_you':
        'The user asks the assistant its own age or when it was created.',
    'make_call':
        'Placing a phone call to a named person.',
    'next_holiday':
        'When the next public holiday or day off falls.',
    'play_music':
        'Starting music playing, whether a song, artist or playlist.',
    'reminder':
        'Reading back the existing reminders. What is on the list.',
    'reminder_update':
        'Creating, changing or deleting a reminder. A write to the list.',
    'shopping_list':
        'Reading back what is on the shopping list.',
    'shopping_list_update':
        'Adding to or removing from the shopping list.',
    'smart_home':
        'Controlling a connected device around the home or car, such as thermostat, air '
        'conditioning or remote start.',
    'todo_list':
        'Reading back what is on the to-do or task list.',
    'todo_list_update':
        'Adding to, clearing or removing from the to-do list.',
    'update_playlist':
        'Adding a song to a playlist or otherwise editing a playlist\'s contents.',
    'what_song':
        'Identifying the song that is currently playing.',
    'next_song':
        'Skipping forward to the next track.',

    # -- auto --------------------------------------------------------------
    'current_location':
        'Where the user is right now.',
    'directions':
        'How to get to a place, or where the nearest one is.',
    'distance':
        'How far away a place is, or how long it takes to reach it.',
    'gas':
        'How much fuel is left in the tank right now.',
    'gas_type':
        'Which grade or kind of fuel the car takes.',
    'jump_start':
        'How to jump start a car with a dead battery.',
    'last_maintenance':
        'When the car was last serviced or had its oil changed. A fact about the past.',
    'mpg':
        'The car\'s fuel economy in miles per gallon.',
    'oil_change_how':
        'The procedure for changing the oil: what to do, or what is needed. A how question.',
    'oil_change_when':
        'When the next oil change is due or how often one is needed. A when question.',
    'schedule_maintenance':
        'Booking a future service, inspection or repair appointment for the car.',
    'tire_change':
        'When tires should be replaced, given when they were last replaced.',
    'tire_pressure':
        'The current pressure in the car\'s tires.',
    'traffic':
        'How heavy traffic is on a route or at a time.',
    'uber':
        'Ordering a rideshare or taxi to a destination.',

    # -- travel ------------------------------------------------------------
    'book_flight':
        'Booking a flight between named places or dates.',
    'book_hotel':
        'Booking or finding a place to stay.',
    'car_rental':
        'Renting a car, or where a car can be rented.',
    'carry_on':
        'Airline carry-on baggage rules, limits and allowances.',
    'flight_status':
        'Whether a flight is on time, or when it lands. Status of a flight that exists.',
    'international_visa':
        'Whether a travel visa is needed for a country.',
    'lost_luggage':
        'Luggage went missing on a flight and the user is reporting or chasing it.',
    'plug_type':
        'Which electrical plug, socket or adapter a country uses.',
    'timezone':
        'Which time zone a named place is in. The zone itself, not the current time.',
    'translate':
        'Rendering a word or phrase into another language.',
    'travel_alert':
        'Whether a destination is safe, or has a warning or advisory against it.',
    'travel_notification':
        'Telling the bank about upcoming travel so the card is not blocked abroad.',
    'travel_suggestion':
        'What to do, see or visit at a destination.',
    'vaccines':
        'Which vaccinations or shots are needed to travel somewhere.',
    'exchange_rate':
        'The conversion rate between two currencies.',

    # -- utility -----------------------------------------------------------
    'alarm':
        'Setting an alarm for a clock time, typically to wake up or be alerted at that time.',
    'calculator':
        'An arithmetic calculation: a percentage, sum, product or fraction.',
    'definition':
        'What a word means.',
    'flip_coin':
        'Flipping a coin.',
    'measurement_conversion':
        'Converting between units of measurement such as inches, centimetres or pounds.',
    'roll_dice':
        'Rolling dice, sometimes with a named number of sides.',
    'share_location':
        'Sending the user\'s location to another person.',
    'spelling':
        'How a word is spelled, or how many of a letter it contains.',
    'text':
        'Sending a text message to someone.',
    'time':
        'What the current clock time is, optionally somewhere else.',
    'timer':
        'Setting a countdown timer for a duration such as ten minutes.',
    'weather':
        'The weather or temperature, now or in the forecast.',

    # -- work --------------------------------------------------------------
    'direct_deposit':
        'Setting up direct deposit of a paycheck into an account.',
    'income':
        'How much the user earns.',
    'insurance':
        'What the user\'s existing insurance or health plan covers or provides.',
    'insurance_change':
        'Switching to a different insurance plan, or starting a new one.',
    'meeting_schedule':
        'Reading back what meetings are already on the schedule.',
    'schedule_meeting':
        'Creating a new meeting, or asking how to create one.',
    'payday':
        'When the user next gets paid, or when they last were.',
    'pto_balance':
        'How much paid time off the user has left or accrued.',
    'pto_request':
        'Making a new request for time off, or asking how to make one.',
    'pto_request_status':
        'Whether a time-off request already submitted has been approved yet.',
    'pto_used':
        'How much paid time off the user has already taken.',
    'rollover_401k':
        'Moving or rolling over a 401k retirement account, usually after changing jobs.',
    'taxes':
        'What the user owes in tax or will get back.',
    'w2':
        'Locating the W-2 tax form.',

    # -- small talk --------------------------------------------------------
    'are_you_a_bot':
        'Whether the assistant is a machine or a person.',
    'do_you_have_pets':
        'Whether the assistant has pets, and what they are.',
    'fun_fact':
        'A piece of trivia or an interesting fact about a named thing.',
    'goodbye':
        'The user is ending the conversation.',
    'greeting':
        'The user is opening the conversation or saying hello.',
    'meaning_of_life':
        'The philosophical question of what life is for.',
    'tell_joke':
        'A request for a joke or something funny.',
    'thank_you':
        'The user is thanking the assistant.',
    'what_are_your_hobbies':
        'What the assistant does for fun or in its spare time.',
    'what_is_your_name':
        'What the assistant is called.',
    'where_are_you_from':
        'Where the assistant comes from or was born.',
    'who_do_you_work_for':
        'Who employs the assistant or who its boss is.',
    'who_made_you':
        'Who built or created the assistant.',

    # -- meta --------------------------------------------------------------
    'cancel':
        'Stop or abandon what is currently happening. An instruction to halt, not an answer '
        'to a question.',
    'change_accent':
        'Switching the assistant\'s voice or accent, such as to a male or British voice.',
    'change_ai_name':
        'Giving the assistant a different name or nickname.',
    'change_language':
        'Switching the language the conversation is conducted in.',
    'change_speed':
        'Making the assistant talk faster or slower.',
    'change_user_name':
        'Telling the assistant what to call the user from now on. Setting the name.',
    'change_volume':
        'Making the assistant louder or quieter, or setting a volume level.',
    'maybe':
        'A hedged or uncertain answer: possibly, not sure, could be either.',
    'no':
        'A negative answer, disagreement, or a statement that something is wrong.',
    'order':
        'Buying or reordering goods, such as restocking something the user has run out of.',
    'order_status':
        'Tracking a package or checking where an existing order has got to.',
    'repeat':
        'The user did not hear and wants the last thing said again.',
    'reset_settings':
        'Returning the assistant or device to its default or factory settings.',
    'sync_device':
        'Pairing, connecting or disconnecting a device such as a phone.',
    'user_name':
        'Asking what the assistant currently calls the user. Reading the name, not setting it.',
    'what_can_i_ask_you':
        'What the assistant is able to help with, or what kinds of questions it takes.',
    'whisper_mode':
        'Switching to whisper mode or asking the assistant to speak softly.',
    'yes':
        'An affirmative answer or agreement.',
}


# ==========================================================================
# MASSIVE (amazon_massive_intent)  --  choice, 60 intents
# ==========================================================================
#
# Source: `mteb/amazon_massive_intent` config `en`, train split. Intent names
# are `scenario_action`. The hard sets are the query/set/remove triples, the
# five hue lighting actions, and play_* versus music_*.

MASSIVE_INSTRUCTIONS = (
    'This is a single spoken command to a smart speaker, transcribed, with numbers written '
    'out as words and the wake word sometimes left in. Which one of the following intents '
    'is the user expressing?'
)

MASSIVE: dict[str, str] = {
    # -- alarm -------------------------------------------------------------
    'alarm_query':
        'Reading back which alarms are already set.',
    'alarm_remove':
        'Deleting or cancelling an alarm.',
    'alarm_set':
        'Creating a new alarm for a time, including "wake me up at" phrasings.',

    # -- audio -------------------------------------------------------------
    'audio_volume_down':
        'Making the sound quieter or reducing the volume.',
    'audio_volume_mute':
        'Silencing the sound entirely, or telling the assistant to be quiet or stop talking.',
    'audio_volume_other':
        'Anything about volume that is not up, down or mute: opening volume settings, or '
        'changing which speaker or how clear the sound is.',
    'audio_volume_up':
        'Making the sound louder, or turning sound back on after it was muted.',

    # -- calendar ----------------------------------------------------------
    'calendar_query':
        'Reading the calendar: what is coming up, what is pending, or what is scheduled.',
    'calendar_remove':
        'Deleting or cancelling an event, meeting or plan.',
    'calendar_set':
        'Creating a calendar entry or a reminder about an event at a time.',

    # -- cooking -----------------------------------------------------------
    'cooking_query':
        'A cooking question that is not a request for a recipe, such as what to cook, or '
        'whether one ingredient can stand in for another.',
    'cooking_recipe':
        'A request for a recipe or for how to cook a named dish.',

    # -- date and time -----------------------------------------------------
    'datetime_convert':
        'Converting a time between two zones, or asking what a time here is there.',
    'datetime_query':
        'Asking what the time or date is, here or in a named place.',

    # -- email -------------------------------------------------------------
    'email_addcontact':
        'Adding or changing an entry in the address book.',
    'email_query':
        'Checking, searching or reading the inbox.',
    'email_querycontact':
        'Looking up someone\'s contact details, such as their phone number or email address.',
    'email_sendemail':
        'Composing, sending or replying to an email.',

    # -- general -----------------------------------------------------------
    'general_greet':
        'Greeting the assistant or opening the conversation.',
    'general_joke':
        'A request for a joke.',
    'general_quirky':
        'A catch-all for anything conversational or open-ended that does not fit another '
        'intent: chit-chat, personal statements, philosophical or odd questions, and stray '
        'web lookups. Use this when the utterance is clearly addressed to the assistant but '
        'names no supported action.',

    # -- iot ---------------------------------------------------------------
    'iot_cleaning':
        'Starting or controlling a robot vacuum or cleaner.',
    'iot_coffee':
        'Making coffee with a connected coffee machine.',
    'iot_hue_lightchange':
        'Changing the colour of the lights, or setting a specific brightness percentage.',
    'iot_hue_lightdim':
        'Making the lights dimmer or less bright.',
    'iot_hue_lightoff':
        'Turning the lights off.',
    'iot_hue_lighton':
        'Turning the lights on.',
    'iot_hue_lightup':
        'Making the lights brighter.',
    'iot_wemo_off':
        'Turning off a smart plug or socket.',
    'iot_wemo_on':
        'Turning on a smart plug or socket, or a device plugged into one.',

    # -- lists -------------------------------------------------------------
    'lists_createoradd':
        'Starting a new list, or adding an item to an existing one.',
    'lists_query':
        'Reading back what is on a list.',
    'lists_remove':
        'Deleting a list, or taking an item off one.',

    # -- music -------------------------------------------------------------
    'music_dislikeness':
        'Expressing dislike for the current track, or asking not to hear it again.',
    'music_likeness':
        'Expressing liking for the current track, saving it or thumbing it up.',
    'music_query':
        'Asking about music without starting playback: what is playing, what is available, '
        'or searching for an artist or album.',
    'music_settings':
        'Changing how playback behaves, such as repeat, shuffle or skipping to another song.',

    # -- news --------------------------------------------------------------
    'news_query':
        'Asking for the news or current headlines.',

    # -- play --------------------------------------------------------------
    'play_audiobook':
        'Starting, resuming or controlling an audiobook.',
    'play_game':
        'Starting or playing a game.',
    'play_music':
        'Starting music playing: a song, artist, genre or station of music.',
    'play_podcasts':
        'Starting, resuming or skipping within a podcast or spoken programme.',
    'play_radio':
        'Starting a radio station, named either by frequency or by name.',

    # -- question answering ------------------------------------------------
    'qa_currency':
        'Exchange rates or what one currency is worth in another.',
    'qa_definition':
        'What a word means.',
    'qa_factoid':
        'A general knowledge question about a person, place or thing. The default for '
        'factual questions that are not currency, maths, stock or definition.',
    'qa_maths':
        'An arithmetic or mathematical calculation.',
    'qa_stock':
        'Share prices or stock market movements.',

    # -- recommendation ----------------------------------------------------
    'recommendation_events':
        'Suggestions for things happening: concerts, festivals, local events.',
    'recommendation_locations':
        'Suggestions for places: restaurants, shops, hotels, somewhere to go.',
    'recommendation_movies':
        'Suggestions about films, including what is showing and when.',

    # -- social ------------------------------------------------------------
    'social_post':
        'Writing or publishing something to social media, including a tweet or status update.',
    'social_query':
        'Reading social media: what is trending, or what is on the user\'s feed.',

    # -- takeaway ----------------------------------------------------------
    'takeaway_order':
        'Placing a food delivery or takeaway order.',
    'takeaway_query':
        'Asking whether a place does delivery or takeaway, or which places do.',

    # -- transport ---------------------------------------------------------
    'transport_query':
        'Directions, routes or public transport timetables.',
    'transport_taxi':
        'Booking a taxi, cab or ride.',
    'transport_ticket':
        'Buying or finding a travel ticket, such as a train or plane ticket.',
    'transport_traffic':
        'How heavy the traffic is, or how long a journey will take because of it.',

    # -- weather -----------------------------------------------------------
    'weather_query':
        'The weather, forecast or temperature.',
}


# ==========================================================================
# AG News  --  choice, 4 topics (the low-cardinality control)
# ==========================================================================
#
# Source: the AG News dataset card's ClassLabel names and the original corpus
# description (news articles grouped into four topics), plus the train split.
# Label keys are the card's names verbatim, including the slash in 'Sci/Tech'.

AG_NEWS_INSTRUCTIONS = (
    'This is the title and opening sentences of a news wire article from around 2004. '
    'Which one of the following four topics does the article belong to?'
)

AG_NEWS: dict[str, str] = {
    'World':
        'International and national news: politics, conflict, diplomacy, disasters and '
        'crime. The default for a story about events in a country rather than about a '
        'company, a team or a technology.',
    'Sports':
        'Sport of any kind: matches, results, players, transfers, championships and '
        'sporting bodies.',
    'Business':
        'Commerce and finance: companies, markets, earnings, oil prices, mergers, jobs and '
        'the economy. Use this when a technology company appears in a story about its '
        'share price, profits or deals rather than its products.',
    'Sci/Tech':
        'Science and technology: research findings, space, the environment, software, '
        'hardware, telecoms and the internet. Use this when the story is about the '
        'technology or the science itself rather than about the money around it.',
}


# ==========================================================================
# SST-5  --  score, 5 ordinal sentiment levels
# ==========================================================================
#
# Source: the Stanford Sentiment Treebank's own five-way scale, whose level
# names the SetFit/sst5 card reproduces as
# 'very positive, positive, neutral, negative, very negative'. The states are
# single sentences from Rotten Tomatoes movie reviews.
#
# Score.criteria is an ordered list, so the gold answer is the level index and
# there is no label string to preserve. Each level therefore quotes the source
# dataset's own `label_text` verbatim at the front of its description, so the
# source surface form is still in the input.

SST5_INSTRUCTIONS = (
    'This is one sentence from a film review. Place the sentiment the writer expresses '
    'about the film on the scale below, from most negative to most positive.'
)

SST5: list[str] = [
    'very negative: the writer condemns the film outright. Dismissive or hostile '
    'throughout, with no redeeming feature offered.',
    'negative: the writer is unfavourable, but measured rather than scathing. Faint '
    'praise, disappointment, or a criticism that stops short of contempt.',
    'neutral: no clear verdict either way. Descriptive, purely factual, genuinely mixed '
    'with the two sides in balance, or an aside that carries no evaluation at all.',
    'positive: the writer is favourable, but measured rather than effusive. Approving '
    'with reservations, or praising one aspect without claiming the film is excellent.',
    'very positive: the writer is enthusiastic and unqualified. Strong praise, '
    'superlatives, or a clear recommendation.',
]


# ==========================================================================
# Civil Comments toxicity  --  noul, binary
# ==========================================================================
#
# Source: the Civil Comments / Jigsaw annotation task documentation. Crowd
# raters were shown a comment and asked to rate its toxicity on the scale
# 'Very Toxic / Toxic / Hard to Say / Not Toxic', where Very Toxic is defined
# as 'a very hateful, aggressive, disrespectful comment or otherwise very
# likely to make a user leave a discussion or give up on sharing their
# perspective'. The released `toxicity` field is the fraction of raters who
# marked the comment toxic; the standard binarisation is >= 0.5.
#
# No test comment was read while writing this. The threshold and its
# consequences are documented in tasks/heldout/README.md.

CIVIL_INSTRUCTIONS = (
    'This is a reader comment posted under a news article. Would the majority of human '
    'moderators rate this comment as toxic, meaning rude, disrespectful, hateful or '
    'aggressive enough that a reasonable person would be likely to leave the discussion '
    'rather than reply to it?'
)

CIVIL: dict[str, str] = {
    'true':
        'A majority of raters would call this toxic: it insults, demeans, threatens or '
        'attacks a person or a group, uses obscenity or slurs aggressively, or is '
        'contemptuous enough to drive people out of the conversation. Strength of opinion '
        'alone does not make a comment toxic; the hostility has to be directed at people.',
    'false':
        'A majority of raters would not call this toxic. This includes blunt, sarcastic, '
        'partisan, angry and strongly worded comments, and criticism of policies, '
        'institutions or public figures\' actions, as long as the comment is not '
        'demeaning or abusive towards people.',
}


# ==========================================================================
# HelpSteer helpfulness  --  score, 5 ordinal levels (non-sentiment ordinal)
# ==========================================================================
#
# Source: the `nvidia/HelpSteer` dataset card, which defines helpfulness as
# 'Overall helpfulness of the response to the prompt' rated by trained human
# annotators on a Likert 5 scale between 0 and 4 where higher is better, plus
# inspection of TRAIN-split examples at each level to fix what the anchors
# mean in practice. The per-level wording in the paper appendix is not
# published in machine-readable form, so these anchors are ours, written from
# the card's definition and the train split, and never from validation.

HELPSTEER_INSTRUCTIONS = (
    'Below is a prompt given to an AI assistant and the response it produced. Rate how '
    'helpful the response is as an answer to that specific prompt. Judge only helpfulness: '
    'a response can be fluent and well written and still be unhelpful if it does not do '
    'what was asked. Ignore how long, how sophisticated or how well styled it is except '
    'where that affects whether the request was met.'
)

HELPSTEER: list[str] = [
    'Not helpful at all. The response does not address the prompt: it refuses, asks for '
    'input that was already supplied, restates the instructions, emits a fragment or '
    'placeholder, or answers a different question entirely.',
    'Barely helpful. It engages with the right topic but largely fails the request: it '
    'misreads what was asked, ignores explicit constraints such as format or length, or '
    'is wrong on most of the substance. A user would have to start over.',
    'Partially helpful. It does some of what was asked and a user could get value from '
    'it, but there are real gaps: part of the request is unaddressed, some content is '
    'wrong, or a stated constraint is only half met. A user would have to follow up.',
    'Helpful. It does what was asked and is substantially correct. Any shortfall is minor: '
    'a small omission, a little padding, or a detail that could be sharper. A user would '
    'be satisfied without needing to ask again.',
    'Completely helpful. It fully satisfies the request, is accurate, and respects the '
    'constraints in the prompt. Nothing important is missing and nothing needs correcting.',
]
