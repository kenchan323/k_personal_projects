from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import selenium.common.exceptions as selenium_exception
import datetime as dt
import pickle
import hotmail

options = webdriver.ChromeOptions()
download_folder = "reports"

profile = {"plugins.plugins_list": [{"enabled": False,
                                     "name": "Chrome PDF Viewer"}],
           "download.default_directory": download_folder,
           "download.extensions_to_open": ""}

options.add_experimental_option("prefs", profile)
driver = webdriver.Chrome("C:\\chromedriver\\chromedriver.exe", chrome_options=options)

class Clothing:

    def __init__(self, name, retailer, url, price=None):
        self.item_name = name
        self.retailer_name = retailer
        self.url = url
        self.price = price
        self.last_check_date = dt.datetime.today()

    def get_current_price(self, driver):
        if self.retailer_name == "levis":
            print(self.url)
            driver.get(self.url)
            while(True):
                try:
                    prices_string = [x.text for x in driver.find_elements_by_class_name("price")]
                except selenium_exception.StaleElementReferenceException:
                    prices_string = [x.text for x in driver.find_elements_by_class_name("price")]

                # Retail price and soft sale price (e.g. temporary?) and hard sale (clearance?)

                prices_string = [x.replace("£", "") for x in prices_string if x != ""]
                prices_string = [x.split(" ") for x in prices_string]
                # Flatten list of list
                prices_string = [item for sublist in prices_string for item in sublist]
                if len(prices_string) != 0:
                  break
            prices_float = [float(x) for x in prices_string]
            prices_float.sort()
                # Get the lowest price and return
            return prices_float[0]


    def check_price_drop(self, driver):
        current_price = self.get_current_price(driver)
        self.last_check_date = dt.datetime.today()
        if (self.price != current_price) and current_price != None:
            old_price = self.price
            self.price = current_price
            return (True, old_price)
        else:
            return (False, self.price)





def get_levis_listings(search_string, driver, pickle_file="C:\\dev\\web_scraper\\levis_510.obj"):
    levis_root = "https://www.levi.com"
    driver.get("https://www.levi.com/GB/en_GB/search/{}".format(search_string))
    listing_elements = driver.find_elements_by_class_name("name")
    listing_elements = [elm for elm in listing_elements if elm.text != ""]
    listing_elements = [elm for elm in listing_elements if (elm.text != "") and "jeans" in elm.text.lower()]
    listing_elements = [elm for elm in listing_elements if not any(y in elm.text.lower() for y in ["boy"])]

    listings = [(elm.text, elm.get_attribute("href")) for elm in listing_elements]

    list_clothe_object = []
    for listing_element in listings:
        print("Processing item #" + str(listings.index(listing_element)))
        print("Out fo a total of " + str(len(listings)) + " items")
        temp_obj = Clothing(listing_element[0], "levis", listing_element[1])
        temp_obj.price = temp_obj.get_current_price(driver)
        list_clothe_object.append(temp_obj)

    print("Done building list fo Clothe objects based on search string " + search_string)

    with open(pickle_file, 'wb') as pickle_file:
        pickle.dump(list_clothe_object, pickle_file)
    print("Done pickling to " + pickle_file)

with open("C:\\Users\\User\\PycharmProjects\\web_scrape_fun\\levis_510.obj", 'rb') as pickle_file:
    list_clothe_object = pickle.load(pickle_file)
print(dt.datetime.today())
email_message = ""
print("Checking for price drops....")
for clothing in list_clothe_object:
    result = clothing.check_price_drop(driver)
    if result[0]:
        email_message = email_message + " \n".join([clothing.item_name, "Was " + str(result[1]), "Now " + str(clothing.price)]) + "\n"
if email_message != "":
    print("FOUND SALES!")
    hotmail.send_email(email_message)
else:
    print("NO SALES FFS!")
driver.close()
print("Checking for price drops....DONE")
print(dt.datetime.today())