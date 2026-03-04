#!/usr/bin/env python3
"""Test Oanda API connection"""

from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountSummary
from oanda_bot.utils.config import Config


def test_oanda_connection():
    """Test connection to Oanda API"""
    try:
        config = Config.load()

        print(f"Testing Oanda API connection...")
        print(f"Environment: {config.oanda.environment}")
        print(f"Account ID: {config.oanda.account_id[:10]}...")

        # Create API client
        api = API(
            access_token=config.oanda.api_token,
            environment=config.oanda.environment
        )

        # Request account summary
        req = AccountSummary(accountID=config.oanda.account_id)
        resp = api.request(req)

        print('\n✓ Oanda API connection successful!')
        print(f"✓ Account Balance: {resp['account']['balance']} {resp['account']['currency']}")
        print(f"✓ Account Type: {resp['account'].get('type', 'N/A')}")

    except Exception as e:
        print(f'\n✗ Oanda API connection failed: {e}')
        print('\nNote: Make sure you have valid Oanda credentials in .env file:')
        print('  - OANDA_ACCOUNT_ID')
        print('  - OANDA_API_TOKEN')
        print('\nGet credentials from: https://www.oanda.com/demo-account/')
        raise


if __name__ == '__main__':
    test_oanda_connection()
