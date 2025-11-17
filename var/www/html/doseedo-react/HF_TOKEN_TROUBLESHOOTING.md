# Hugging Face Token Troubleshooting

## Issue: Token Authentication Failed (401 Unauthorized)

Your token `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` is returning "Invalid credentials".

### Possible Causes:

1. **Token has been revoked or expired**
2. **Token was copied incorrectly** (extra spaces, missing characters)
3. **Token doesn't have proper permissions**
4. **Account verification required**

---

## Solution: Generate a New Token

### Step 1: Go to Hugging Face Settings

1. Visit: https://huggingface.co/settings/tokens
2. Sign in to your account

### Step 2: Create a New Token

1. Click **"New token"** or **"Create new token"**
2. Give it a name: `doseedo-app` or similar
3. **Important**: Select permissions:
   - ✅ **Read** (minimum required)
   - ✅ **Write** (if you want to upload/modify)
   - For API calls, "Read" is sufficient
4. Click **"Generate"**
5. **Copy the token immediately** (you won't see it again!)

### Step 3: Update Your .env File

1. Open: `/var/www/html/doseedo-react/.env`
2. Replace the token:
   ```bash
   REACT_APP_HF_API_TOKEN=hf_your_new_token_here
   ```
3. Save the file

### Step 4: Test the New Token

Run this command to verify:
```bash
cd /var/www/html/doseedo-react
node test-hf-connection.js
```

You should see:
```
✅ Token is valid!
   User: your-username
   Type: user
```

---

## Alternative: Check Current Token Status

Visit your tokens page and check if the token:
- Shows as "Active" (green)
- Has not expired
- Has the right permissions

If the token shows as "Revoked" or "Expired", you must create a new one.

---

## Quick Fix Commands

Once you have a new valid token, run:

```bash
# Update .env file (replace YOUR_NEW_TOKEN)
echo 'REACT_APP_HF_API_TOKEN=YOUR_NEW_TOKEN' > /var/www/html/doseedo-react/.env

# Test connection
cd /var/www/html/doseedo-react && node test-hf-connection.js

# Rebuild app
npm run build && sudo cp -r build/* /var/www/html/
```

---

## Need Help?

If you continue having issues:
1. Make sure you're signed in to the correct HF account
2. Check if your account needs email verification
3. Try logging out and back in to Hugging Face
4. Contact Hugging Face support if the issue persists

---

**Important Security Note**: Never share your token publicly or commit it to git!
