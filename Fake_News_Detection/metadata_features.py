from six import string_types

COLUMNS_NAMES = ["id", 
                "label", "statement", "subject", "speaker", "job", "state", "party",  # columns 1-8
                "barely-true-counts", "false-counts", "half-true-counts", "mostly-true-counts", "pants-fire-counts",  # columns 9-13 (counts)
                "venue"]  # names of the columns in the tsv files

# Map output label to six classes.
LABEL_LIST_REVERSE = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL_DICT = {"pants-fire":0, "false":0, "barely-true":0, "half-true":1, "mostly-true":1, "true":1}
SPEAKERS_LIST = ["barack-obama", "donald-trump", "hillary-clinton", "mitt-romney", 
            "scott-walker", "john-mccain", "rick-perry", "chain-email", 
            "marco-rubio", "rick-scott", "ted-cruz", "bernie-s", "chris-christie", 
            "facebook-posts", "charlie-crist", "newt-gingrich", "jeb-bush", 
            "joe-biden", "blog-posting","paul-ryan"]
JOB_LIST = ["president", "u.s. senator", "governor", "president-elect", "presidential candidate", 
            "u.s. representative", "state senator", "attorney", "state representative", "congress"]
JOB_DICT = {"president":0, "u.s. senator":1, "governor":2, "president-elect":3, "presidential candidate":4, 
            "u.s. representative":5, "state senator":6, "attorney":7, "state representative":8, "congress":9}
PARTY_DICT = {"republican":0,"democrat":1,"none":2,"organization":3,"newsmaker":4}    

# Possible groupings (50 groups + 1 for rest).
STATES_DICT = {"wyoming": 48, "colorado": 5, "washington": 45, "hawaii": 10, "tennessee": 40, 
               "wisconsin": 47, "nevada": 26, "north dakota": 32, "mississippi": 22, "south dakota": 39, 
               "new jersey": 28, "oklahoma": 34, "delaware": 7, "minnesota": 21, "north carolina": 31, 
               "illinois": 12, "new york": 30, "arkansas": 3, "west virginia": 46, "indiana": 13, 
               "louisiana": 17, "idaho": 11, "south  carolina": 38, "arizona": 2, "iowa": 14, "maine":49, "maryland": 18,
               "michigan": 20, "kansas": 15, "utah": 42, "virginia": 44, "oregon": 35, "connecticut": 6, 
               "montana": 24, "california": 4, "massachusetts": 19, "rhode island": 37, "vermont": 43, 
               "georgia": 9, "pennsylvania": 36, "florida": 8, "alaska": 1, "kentucky": 16, "nebraska": 25, 
               "new hampshire": 27, "texas": 41, "missouri": 23, "ohio": 33, "alabama": 0, "new mexico": 29}

# Possible groups (14).
SUBJECT_LIST = ["health","tax","immigration","election","education",
                "candidates-biography","economy","gun","jobs","federal-budget","energy","abortion","foreign-policy"]
SUBJECT_DICT = {"health":0,"tax":1,"immigration":2,"election":3,"education":4,
                "candidates-biography":5,"economy":6,"gun":7,"jobs":8,"federal-budget":9,"energy":10,"abortion":11,"foreign-policy":12}

VENUE_LIST = ["news release","interview","tv","radio",
              "campaign","news conference","press conference","press release",
              "tweet","facebook","email"]
VENUE_DICT = {"news release":0,"interview":1,"tv":2,"radio":3,
              "campaign":4,"news conference":5,"press conference":6,"press release":7,
              "tweet":8,"facebook":9,"email":10}

###########################
# Map Features  
###########################

def map_speaker(speaker):
    speaker_dict = {}
    for cnt, speaker in enumerate(SPEAKERS_LIST):
        speaker_dict[speaker] = cnt
    # print(speaker_dict)
    # print(len(SPEAKERS_LIST))

    if isinstance(speaker, string_types):
        speaker = speaker.lower()
        matches = [s for s in SPEAKERS_LIST if s in speaker]
        if len(matches) > 0:
            return speaker_dict[matches[0]] #Return index of first match
        else:
            return len(SPEAKERS_LIST)
    else:
        return len(SPEAKERS_LIST) #Nans or un-string data goes here.
 
def map_job(job):
    """ Possible groupings could be (11 groups):
    # president, us senator, governor(contains governor), president-elect, presidential candidate, us representative,
    # state senator, attorney, state representative, congress (contains congressman or congresswoman), rest
    """
    if isinstance(job, string_types):
        job = job.lower()
        matches = [s for s in JOB_LIST if s in job]
        if len(matches) > 0:
            return JOB_DICT[matches[0]] #Return index of first match
        else:
            return 10 #This maps any other job to index 10
    else:
        return 10 #Nans or un-string data goes here.

def map_party(party):
    if party in PARTY_DICT:
        return PARTY_DICT[party]
    else:
        return 5 # default index for rest party is 5

def map_state(state):
    if isinstance(state, string_types):
        state = state.lower()  # convert to lowercase
        if state in STATES_DICT:
            return STATES_DICT[state]
        else:
            return 50 # This maps any other location to index 50
    else:
        return 50 # Nans or un-string data goes here.

#possibe groups (12)
#news release, interview, tv (television), radio, campaign, news conference, press conference, press release,
#tweet, facebook, email, rest
def map_venue(venue):
    if isinstance(venue, string_types):
        venue = venue.lower()
        matches = [s for s in VENUE_LIST if s in venue]
        if len(matches) > 0:
            return VENUE_DICT[matches[0]] #Return index of first match
        else:
            return 11 #This maps any other venue to index 11
    else:
        return 11 # Nans or un-string data goes here.

#health-care,taxes,immigration,elections,education,candidates-biography,guns,
#economy&jobs ,federal-budget,energy,abortion,foreign-policy,state-budget, rest
#Economy & Jobs is bundled together, because it occurs together
def map_subject(subject):
    if isinstance(subject, string_types):
        subject = subject.lower()
        matches = [s for s in SUBJECT_LIST if s in subject]
        if len(matches) > 0:
            return SUBJECT_DICT[matches[0]] #Return index of first match
        else:
            return 13 # This maps any other subject to index 13
    else:
        return 13 # Nans or un-string data goes here.