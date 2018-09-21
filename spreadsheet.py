import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope= ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds=ServiceAccountCredentials.from_json_keyfile_name('attendance-b4a9012c7c2d.json',scope)
client=gspread.authorize(creds)
sheet=client.open('attendance').sheet1

attendance = sheet.get_all_records()

print(attendance)
