import urllib3
import urllib
import json
# Suppress url connect safety warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FSA_ROOT = "https://ratings.food.gov.uk"
def get_council_id(council):
    http = urllib3.PoolManager()
    request = http.request('GET', "https://ratings.food.gov.uk/authorities/json")
    results = json.loads(request.data.decode('utf8'))
    results = results["ArrayOfWebLocalAuthorityAPI"]
    results = results['WebLocalAuthorityAPI']
    for result in results:
        if council in result["Name"]:
            return result['LocalAuthorityIdCode']

def get_fhrs_info(restaurant_name, postcode="^", loc_auth="London", ret_multiple=True):
    # ASCII encoding
    restaurant = urllib.parse.quote(restaurant_name)
    postcode = urllib.parse.quote(postcode)
    loc_auth = urllib.parse.quote(loc_auth)
    # See syntax in https://ratings.food.gov.uk/open-data-resources/documents/api-guidance.pdf
    url = FSA_ROOT + "/enhanced-search/en-GB/{}/{}/alpha/0/{}/%5E/1/1/10/json".format(restaurant, postcode, loc_auth)
    # {lang} / {name} / {address} / {sortOrder} / {businessType} / {la} / {page} / {pageSize} / {format}

    http = urllib3.PoolManager()
    request = http.request('GET', url)
    results = json.loads(request.data.decode('utf8'))
    results = results["FHRSEstablishment"]
    if not "EstablishmentCollection" in results.keys():
        # No restaurant found
        return None

    results = results["EstablishmentCollection"]['EstablishmentDetail']

    if type(results) == dict:
        # Great ! We only had one result returned
        return results

    elif type(results) == list:   # If returned multiple results, we loop through to look for the exact match

        if ret_multiple:
            # We got multiple results and user wants to see all, we return all
            return results

        for res in results:
            # Otherwise we loop through all results and return the first one that has the identical name
            if res["BusinessName"] == restaurant_name:
                return res

        return None


# get_fhrs_info("KFC", postcode = "E16 4HQ")
# di_res = get_fhrs_info("KFC", postcode="E16 4HQ", loc_auth="Newham")
