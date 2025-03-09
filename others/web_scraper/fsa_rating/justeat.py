from selenium import webdriver
import pandas as pd
import fsa_rating

import os
import datetime as dt

options = webdriver.ChromeOptions()
download_folder = "reports"

profile = {"plugins.plugins_list": [{"enabled": False,
                                     "name": "Chrome PDF Viewer"}],
           "download.default_directory": download_folder,
           "download.extensions_to_open": ""}

options.add_experimental_option("prefs", profile)
driver = webdriver.Chrome("C:\\chromedriver\\chromedriver.exe", chrome_options=options)
# driver.get("http://cottonhost.com/96726/")
'''
driver.get("http://google.com")
actions = ActionChains(driver)
driver.back()
'''


# Input the search string
# search_input = driver.find_element_by_xpath("/html/body/div/div[3]/form/div[2]/div/div[1]/div/div[1]/input").send_keys("Ken Chan")
# # Click on a blank space to hide the search suggestion box
# driver.find_element_by_xpath("//body").click()
# # Click on the search button
# element = driver.find_element_by_xpath("/html/body/div/div[3]/form/div[2]/div/div[3]/center/input[1]").click()

driver.get("https://www.just-eat.co.uk")
search_input = driver.find_element_by_name("postcode").send_keys("E14 0LR")
element = driver.find_element_by_xpath("/html/body/app/div/div/div[1]/div[2]/form/div/button").click()
# Get all restaurant names
listing_elements = driver.find_elements_by_class_name("c-listing-item-title")
list_restaurant = [x.text for x in listing_elements if x.text != ""]
print("There are a total of {} restaurants".format(len(list_restaurant)))
driver.quit()
# Now we gather the rating of all of these restaurants
options.add_experimental_option("prefs", profile)

results_df = pd.DataFrame(columns=["Name", "Postcode", "Rating", "Rating_Date", "Council"])

councils = ["Tower Hamlets", "Newham"]
council_ids = [fsa_rating.get_council_id(x) for x in councils]

for restaurant in list_restaurant:
    print("Fetching for {}...".format(restaurant))
    for council in council_ids:
        fsa_result = fsa_rating.get_fhrs_info(restaurant, loc_auth=council, ret_multiple=False)
        if fsa_result != None:
            break
    if fsa_result == None:
        continue
    dict_temp = {}
    dict_temp["Name"] = restaurant
    dict_temp["Postcode"] = fsa_result["PostCode"]
    dict_temp["Rating"] = fsa_result["RatingValue"]
    dict_temp["Rating_Date"] = dt.datetime.strptime(fsa_result["RatingDate"], "%d %B %Y")
    dict_temp["Council"] = fsa_result["LocalAuthorityName"]

    results_df = results_df.append(dict_temp, ignore_index=True)
    print("Fetching for {}...DONE".format(restaurant))

results_df.to_csv(os.path.join("output", "restaurant.csv"), index=False)

# driver.get("https://ratings.food.gov.uk/")
#
# council_names = ["Tower Hamlets", "Newham"]
# for restaurant_name in list_restaurant[14:]:
#     try:
#         found_restaurant = False
#         clear_restaurant_name_box = driver.find_element_by_name("ctl00$MainContent$uxSearchBox$uxSearchName").clear()
#         enter_restaurant_name = driver.find_element_by_name("ctl00$MainContent$uxSearchBox$uxSearchName").send_keys(restaurant_name)
#
#         for council in council_names:
#             try:
#                 expand_search_option = driver.find_element_by_class_name("showMoreArrow").click()
#             except selenium_exception.ElementNotVisibleException:
#                 print("Okay I can't find the showMoreArrow button...")
#             council_selector = Select(driver.find_element_by_id("MainContent_uxSearchBox_uxSearchLocalAuthority"))
#             council_selector.select_by_visible_text(council)
#
#             # Search!
#             click_search_button = driver.find_element_by_id("MainContent_uxSearchBox_uxSearchAction").click()
#             time.sleep(2)
#             if len(driver.find_elements_by_id("MainContent_noResultsWrapper")) == 1:
#                 found_restaurant = False
#                 # We need to change the council to try again
#                 driver.back()
#             else:
#                 found_restaurant = True
#                 dict_restaurant_res = {}
#                 dict_restaurant_res["Name"] = restaurant_name
#                 # Found a (some) restaurants, we shall just look at the first result
#                 try:
#                     restaurant_result_row = driver.find_elements_by_class_name("ResultRow")[0]
#                 except IndexError:
#                     print("Index Error as we haven't waited long enough after the Search click")
#                     found_restaurant = False
#                     continue
#                 # Get restaurant postcode
#                 restaurant_result_pc = restaurant_result_row.find_elements_by_class_name("ResultsBusinessPostcode")[0].text
#                 dict_restaurant_res["Postcode"] = restaurant_result_pc
#                 # Now try to get rating
#                 rating_element = restaurant_result_row.find_elements_by_class_name("ratingColumnPadding")[0]
#                 rating_element_img = rating_element.find_elements_by_tag_name("img")[0]
#                 rating_string = rating_element_img.get_property("title").replace("'", "").replace(":","")
#                 if rating_string == "Awaiting inspection":
#                     dict_restaurant_res["Rating"] = -1
#                     dict_restaurant_res["Rating_Date"] = dt.datetime.today()
#                     driver.back()
#                     results_df = results_df.append(dict_restaurant_res, ignore_index=True)
#                     break
#                 if rating_string == "Food Hygiene Rating Scheme Exempt":
#                     dict_restaurant_res["Rating"] = -2
#                     dict_restaurant_res["Rating_Date"] = dt.datetime.today()
#                     driver.back()
#                     results_df = results_df.append(dict_restaurant_res, ignore_index=True)
#                     break
#                 rating_string_tokens = rating_string.split(" ")
#                 rating_int = [int(x) for x in rating_string_tokens if x.isnumeric()][0]
#                 dict_restaurant_res["Rating"] = rating_int
#                 # Now get rating date
#                 rating_date_element = restaurant_result_row.find_elements_by_class_name("ResultsRatingDate")[0]
#                 rating_date = rating_date_element.find_element_by_id("SearchResults_uxSearchResults_EstablishmentRatingDateNoLocation_0").text
#                 dict_restaurant_res["Rating_Date"] = pd.to_datetime(rating_date)
#                 dict_restaurant_res["Council"] = council
#                 driver.back()
#
#             if found_restaurant:
#                 results_df = results_df.append(dict_restaurant_res, ignore_index=True)
#                 break
#
#     except selenium_exception.ElementClickInterceptedException:
#         print("Click Intercepted Error On : " + restaurant_name)
