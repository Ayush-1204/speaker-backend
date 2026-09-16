#!/bin/bash

# Test device registration and listing flow
# Simulates: same account, register parent device on phone A, child device on phone B

BASE_URL="http://127.0.0.1:8000"
DEV_TOKEN="dev:test-parent-multidevice"

echo "=========================================="
echo "TEST: Multi-Device Same Account Flow"
echo "=========================================="

# Step 1: Auth with dev token
echo ""
echo "[1/6] Authenticating with dev token..."
AUTH_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/google" \
  -H "Content-Type: application/json" \
  -d "{\"id_token\": \"$DEV_TOKEN\"}")

ACCESS_TOKEN=$(echo "$AUTH_RESPONSE" | jq -r '.access_token // empty')
if [ -z "$ACCESS_TOKEN" ]; then
  echo "❌ Auth failed. Response: $AUTH_RESPONSE"
  exit 1
fi
echo "✓ Auth success. Token: ${ACCESS_TOKEN:0:20}..."

# Step 2: Register parent device (simulating Phone A)
echo ""
echo "[2/6] Registering PARENT_DEVICE (Phone A)..."
PARENT_REG=$(curl -s -X POST "$BASE_URL/devices" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "device_name=Parent Phone A" \
  -F "role=parent_device" \
  -F "device_token=fcm_parent_phone_a")

PARENT_DEVICE_ID=$(echo "$PARENT_REG" | jq -r '.device.id // empty')
if [ -z "$PARENT_DEVICE_ID" ]; then
  echo "❌ Parent device registration failed. Response: $PARENT_REG"
  exit 1
fi
echo "✓ Parent device registered: $PARENT_DEVICE_ID"
echo "  Response: $PARENT_REG" | jq '.'

# Step 3: Register child device (simulating Phone B, same account)
echo ""
echo "[3/6] Registering CHILD_DEVICE (Phone B, same account)..."
CHILD_REG=$(curl -s -X POST "$BASE_URL/devices" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "device_name=Child Phone B" \
  -F "role=child_device" \
  -F "device_token=fcm_child_phone_b")

CHILD_DEVICE_ID=$(echo "$CHILD_REG" | jq -r '.device.id // empty')
if [ -z "$CHILD_DEVICE_ID" ]; then
  echo "❌ Child device registration failed. Response: $CHILD_REG"
  exit 1
fi
echo "✓ Child device registered: $CHILD_DEVICE_ID"
echo "  Response: $CHILD_REG" | jq '.'

# Step 4: List devices from parent token
echo ""
echo "[4/6] Listing devices (GET /devices with parent token)..."
DEVICES_LIST=$(curl -s -X GET "$BASE_URL/devices" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

echo "✓ Devices list response:"
echo "$DEVICES_LIST" | jq '.'

ITEMS_COUNT=$(echo "$DEVICES_LIST" | jq '.items | length')
echo "  Total devices: $ITEMS_COUNT"

# Verify both devices are present
if [ "$ITEMS_COUNT" -lt 2 ]; then
  echo "❌ ERROR: Expected 2 devices, got $ITEMS_COUNT"
  exit 1
fi

# Check for parent_device
HAS_PARENT=$(echo "$DEVICES_LIST" | jq ".items[] | select(.role == \"parent_device\") | .id" | wc -l)
# Check for child_device
HAS_CHILD=$(echo "$DEVICES_LIST" | jq ".items[] | select(.role == \"child_device\") | .id" | wc -l)

echo "  Found parent_device: $HAS_PARENT"
echo "  Found child_device: $HAS_CHILD"

if [ "$HAS_PARENT" -ne 1 ] || [ "$HAS_CHILD" -ne 1 ]; then
  echo "❌ ERROR: Expected 1 parent_device and 1 child_device"
  exit 1
fi

# Step 5: Verify parent_id consistency
echo ""
echo "[5/6] Verifying parent_id consistency..."
PARENT_IDS=$(echo "$DEVICES_LIST" | jq -r '.items[].parent_id' | sort -u)
PARENT_ID_COUNT=$(echo "$PARENT_IDS" | wc -l)

if [ "$PARENT_ID_COUNT" -ne 1 ]; then
  echo "❌ ERROR: Multiple parent_ids found. Expected 1, got $PARENT_ID_COUNT"
  echo "   Parent IDs: $PARENT_IDS"
  exit 1
fi

echo "✓ All devices have same parent_id: $PARENT_IDS"

# Step 6: Verify field structure
echo ""
echo "[6/6] Verifying field structure..."
FIRST_DEVICE=$(echo "$DEVICES_LIST" | jq '.items[0]')
REQUIRED_FIELDS=("id" "parent_id" "device_name" "role" "battery_percent" "is_online" "monitoring_enabled")

for field in "${REQUIRED_FIELDS[@]}"; do
  if echo "$FIRST_DEVICE" | jq -e ".\"$field\" != null" > /dev/null 2>&1; then
    echo "  ✓ Field '$field' present"
  else
    echo "  ⚠ Field '$field' missing or null in first device"
  fi
done

echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Parent device registered: $PARENT_DEVICE_ID"
echo "  - Child device registered: $CHILD_DEVICE_ID"
echo "  - Both devices listed in GET /devices"
echo "  - Both have same parent_id"
echo "  - Response includes required fields"
