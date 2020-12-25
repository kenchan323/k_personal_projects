import time

from selenium import webdriver
import pandas as pd

'''
Thi script uses Selenium with Chrome driver to scrape all the trades submitted by current and former US senators. This 
will only include reported trades that were submitted in a structured manner (e.g. not a print out of trade confirmation
by brokers).

@kenchan323
2020-12-20
'''

def _get_current_page_count(chrome_driver):
    '''
    Given a chrome driver object that currently points to a "Search Result" page on the US Senate Financial Disclosure
    page, return the current page number
    :param chrome_driver: Chrome webdriver object
    :return: int - current page number
    '''
    elm_current_page = chrome_driver.find_elements_by_class_name("paginate_button.current")[0]
    curr_page_count = int(elm_current_page.text)
    return curr_page_count

def _click_next_page(chrome_driver, curr_page_count):
    '''
    Given a chrome driver object that currently points to a "Search Result" page on the US Senate Financial Disclosure
    page, turn to the next page
    :param chrome_driver: Chrome webdriver object
    :param curr_page_count: int - current page count
    :return:
    '''
    # Time to move to next page
    list_elm_next_page = chrome_driver.find_elements_by_class_name("paginate_button")
    for elm_button in list_elm_next_page:
        if not str(elm_button.text).isnumeric():
            # maybe "Previous" or "Next"
            continue
        if int(elm_button.text) == curr_page_count + 1:
            elm_button.click()
            print(f"Moved to page {elm_button.text}")
            time.sleep(1)
            break

def _open_doc_page_parse(url, **kwargs):
    '''
    Given a submission level url on the Senate Financial Disclosure website, check if there is a structured table with
    the submitted trade level information and if so, return the data in the form of a list of dict objects
    :param url: str - url of the submission level
    :param kwargs: additional information to be added to each dict object (e.g. Senator="Mike Pence")
    :return: list - list of dict object
    '''
    list_dict_out = []
    # Open in new tab
    driver.execute_script(f"window.open('{url}');")
    # Activate the new tab
    driver.switch_to.window(driver.window_handles[1])

    list_elm_tables = driver.find_elements_by_class_name("card-body")
    if len(list_elm_tables) != 1:
        print("No table found")
        # Close the new tab
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        return None

    elm_table = list_elm_tables[0]
    list_headers = [x.text for x in elm_table.find_element_by_class_name("header").find_elements_by_tag_name("th")]
    list_rows = elm_table.find_elements_by_xpath("//tr")
    for row_elm in list_rows:
        list_field_elm = row_elm.find_elements_by_tag_name("td")
        list_val = [x.text for x in list_field_elm]
        dict_key_val = {list_headers[idx]: v for idx, v in enumerate(list_val)}
        if len(dict_key_val) == 0:
            # no data parsed here, so we just continue to next row
            continue
        # The additional kwargs we had supplied (e.g. person's name)
        for k, v in kwargs.items():
            dict_key_val[k] = v

        list_dict_out.append(dict_key_val)
    # Close this tab
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return list_dict_out

PATH_CHROME_DRIVER = "C:\\chromedriver\\chromedriver.exe"
URL_SENATE_DISCLOSURE = "https://efdsearch.senate.gov/search/"

str_earliest_search_dt = "12/24/2012"
options = webdriver.ChromeOptions()
download_folder = "reports"

profile = {"plugins.plugins_list": [{"enabled": False,
                                     "name": "Chrome PDF Viewer"}],
           "download.default_directory": download_folder,
           "download.extensions_to_open": ""}

options.add_experimental_option("prefs", profile)
driver = webdriver.Chrome(PATH_CHROME_DRIVER, chrome_options=options)

driver.get(URL_SENATE_DISCLOSURE)

# Now need to take check box for agreement
elm_agree_statement = driver.find_element_by_id("agree_statement")
elm_agree_statement.click()

elm_start_date_search_in = driver.find_element_by_id("fromDate")
elm_start_date_search_in.send_keys(str_earliest_search_dt)
elm_start_date_search_in.send_keys()

# Only want to search "Periodic Transaction"
elm_periodic_trans = driver.find_elements_by_id("reportTypes")[1]
elm_periodic_trans.click()
# Now we click the search button
elm_search_button = driver.find_elements_by_class_name("btn-primary")[0]
elm_search_button.click()

time.sleep(2) # let the page load
# Find max number of page (last page button)
elm_final_page = driver.find_elements_by_class_name("paginate_button")[-2]
final_page_count = int(elm_final_page.text)
print(f"Total {final_page_count} pages to crawl through!")

curr_page_count = _get_current_page_count(driver)

list_d = []
while curr_page_count <= final_page_count:
    at_first_page = (curr_page_count == 1)
    if not at_first_page:
        print(f"About to move to Page {curr_page_count + 1} .....")
        _click_next_page(driver, curr_page_count)
        curr_page_count = curr_page_count + 1

    # Get current page count
    curr_page_count = _get_current_page_count(driver)
    # First element is skipped because those are the headers
    list_row_elms = driver.find_elements_by_xpath(" //tr[@role = 'row']")[1:]
    # 3 because link is embedded in the third column
    list_doc_links = [x.find_elements_by_tag_name("td")[3].find_element_by_tag_name("a").get_attribute("href")
                      for x in list_row_elms]
    list_first_name = [x.find_elements_by_tag_name("td")[0].text for x in list_row_elms]
    list_last_name = [x.find_elements_by_tag_name("td")[1].text for x in list_row_elms]
    list_full_name = [x + "," + y for x,y in zip(list_first_name, list_last_name)]
    # sub_link = 'https://efdsearch.senate.gov/search/view/ptr/05f5d46b-c3b0-4ef3-aaa6-a1f49574b5af/'
    # page = requests.get(sub_link)
    # soup = BeautifulSoup(page.content, 'html.parser')
    for idx, link in enumerate(list_doc_links):
        person = list_full_name[idx]
        ret_dict = _open_doc_page_parse(link, Senator=person)
        if ret_dict == None:
            continue
        list_d.extend(ret_dict)

# We loop through all the dict objects in the list and append them into a pandas DataFrame
df = pd.DataFrame(columns=list_d[0].keys())
for dict_entry in list_d:
    df = df.append(dict_entry, ignore_index=True)
df.to_csv("scraped_senate_trades.csv")