from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException


def init_drive(url):
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(20)
    try:
        driver.get(url)
    except TimeoutException:
        pass
    return driver


def search_stock(driver, stock_name):
    search_box = driver.find_element_by_id('search_str')
    search_box.send_keys(stock_name)
    submit = driver.find_element_by_class_name("btn_search")
    try:
        submit.click()
    except TimeoutException:
        pass
    try:
        stock = driver.find_elements_by_class_name('bl_18')
        stock[-1].click()
    except NoSuchElementException:
        pass
    except IndexError:
        pass
    driver = click_news(driver, stock_name)
    return driver


def click_news(driver, stock_name):
    driver.implicitly_wait(10)
    news_button = driver.find_element_by_xpath('//*[@id="slider"]/dt[3]/a')
    news_button.click()
    with open("search_urls.txt", 'r') as f:
        lines = f.readlines()
        lines = [l.replace('\n', '') for l in lines]
        print(lines)

    if stock_name not in lines:
        with open("search_urls.txt", 'w') as f:
            lines.append(stock_name)
            lines.append(url)
            for i, l in enumerate(lines):
                if i == 1:
                    f.write(l)
                else:
                    f.write('\n' + l)
    return driver


if __name__ == '__main__':
    snames = ['']
    url = 'https://www.moneycontrol.com/'
    stock_name = 'Reliance Industries'
    driver = init_drive(url)
    driver = search_stock(driver, stock_name)
