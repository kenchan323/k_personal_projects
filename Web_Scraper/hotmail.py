from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(message):
    s = smtplib.SMTP('smtp-mail.outlook.com', 587) #Change smtp for Outlook
    s.starttls()
    s.login("kencheckprice@hotmail.com", "checkprice323")

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # Prints out the message body for our sake
    # print(message)

    # setup the parameters of the message
    msg['From']= "kencheckprice@hotmail.com"
    msg['To']='kenchan323@hotmail.com'
    msg['Subject']="Test"

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    # send the message via the server set up earlier.
    s.send_message(msg)

    # Terminate the SMTP session and close the connection
    s.quit()