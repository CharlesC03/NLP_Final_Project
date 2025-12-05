import http.client, urllib
import requests

from config import user_key, token_key
from typing import Dict

class Notifier:
    def __init__(self, enabled: bool = False):
        """
        Initialize the Notifier instance.
        Args:
            enabled (bool, optional): Whether notifications are enabled. Defaults to False.
        Attributes:
            conn (http.client.HTTPSConnection): HTTPS connection to Pushover API server.
            enabled (bool): Flag indicating if notifications are active.
        """
        self.r = http.client.HTTPSConnection("api.pushover.net:443")
        self.enabled = enabled
    
    def set_usage(self, enabled: bool):
        """
        Set whether the notifier is enabled or disabled.
        Args:
            enabled (bool): True to enable notifications, False to disable them.
        """

        self.enabled = enabled

    def send_notification(self, message: str, additional_info: Dict[str, str] = {}, title: str = "NLP Project"):
        """
        Send a push notification via Pushover API.
        This method sends a notification message to a user through the Pushover service using 
        the configured token and user keys.
        Args:
            message (str): The main notification message content to be sent.
            additional_info (Dict[str, str], optional): Additional parameters to include in the 
                notification (e.g., priority, sound, url). Defaults to empty dict.
            title (str, optional): The title of the notification. Defaults to "NLP Project".
        Returns:
            None
        Raises:
            Prints an error message to stdout if the API request fails (non-200 status code).
        Note:
            Requires `self.conn`, `token_key`, and `user_key` to be defined in the class scope.
            The notification is sent as a POST request to the Pushover API endpoint.
        """
        if not self.enabled:
            return
        self.conn.request("POST", "/1/messages.json",
            urllib.parse.urlencode({
                "token": token_key,
                "user": user_key,
                "title": title,
                "message": message,
                **additional_info,
            }), { "Content-type": "application/x-www-form-urlencoded" })
        res = self.conn.getresponse()
        if res.status != 200:
            print(f"Failed to send notification: {res.status} {res.reason}")
    
    def send_image(self, message: str, image_path: str, additional_info: Dict[str, str] = {}, title: str = "NLP Project"):
        """
        Sends a push notification with an image attachment via the Pushover API.
        This method posts a message with an attached image file to the configured Pushover
        user account. If notifications are disabled (self.enabled is False), the method
        returns immediately without sending.
        Args:
            None
        Returns:
            None
        Side Effects:
            - Sends an HTTP POST request to the Pushover API
            - Opens and reads an image file from disk
            - Prints an error message to stdout if the API request fails
        Raises:
            May raise exceptions related to file I/O or network requests if the image
            file doesn't exist or network connectivity issues occur.
        Note:
            - Requires valid 'token_key' and 'user_key' to be configured
            - Currently hardcoded to send "path_to_image.jpg"
            - Does not close the file handle explicitly (relies on garbage collection)
        """
        if not self.enabled:
            return
        url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": token_key,
            "user": user_key,
            "title": title,
            "message": message,
            **additional_info,
        }
        files = {
            "attachment": open(image_path, "rb"),
        }
        response = requests.post(url, data=data, files=files)
        if response.status_code != 200:
            print(f"Failed to send image notification: {response.status_code} {response.reason}")