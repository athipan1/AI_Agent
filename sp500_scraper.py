import requests
from bs4 import BeautifulSoup
import csv


def scrape_sp500_companies():
    """
    Scrapes the list of S&P 500 companies from Wikipedia and saves it to a CSV file.
    The CSV file will contain Symbol, Security, GICS Sector, and GICS Sub-Industry.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/58.0.3029.110 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # The table with the components has the id 'constituents'
    table = soup.find('table', {'id': 'constituents'})

    if not table:
        print("Could not find the S&P 500 components table.")
        return

    companies = []
    # The first row is the header, so we skip it
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) > 3:
            symbol = cols[0].text.strip()
            security = cols[1].text.strip()
            gics_sector = cols[2].text.strip()
            gics_sub_industry = cols[3].text.strip()
            companies.append([symbol, security, gics_sector, gics_sub_industry])

    # Write the data to a CSV file
    try:
        with open('sp500_companies.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry'])
            writer.writerows(companies)
        print("Successfully saved S&P 500 data to sp500_companies.csv")
    except IOError as e:
        print(f"Error writing to file: {e}")


if __name__ == '__main__':
    scrape_sp500_companies()
