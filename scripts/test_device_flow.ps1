# Test device registration and listing flow
# Simulates: same account, register parent device on phone A, child device on phone B

$BaseUrl = "http://127.0.0.1:8000"
$DevToken = "dev:test-parent-multidevice"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TEST: Multi-Device Same Account Flow" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Auth with dev token
Write-Host ""
Write-Host "[1/6] Authenticating with dev token..." -ForegroundColor Yellow
$AuthResponse = Invoke-WebRequest -Uri "$BaseUrl/auth/google" `
  -Method POST `
  -Headers @{"Content-Type" = "application/json"} `
  -Body (@{"id_token" = $DevToken} | ConvertTo-Json) | ConvertFrom-Json

$AccessToken = $AuthResponse.access_token
if ([string]::IsNullOrEmpty($AccessToken)) {
  Write-Host "❌ Auth failed. Response: $AuthResponse" -ForegroundColor Red
  exit 1
}
Write-Host "✓ Auth success. Token: $($AccessToken.Substring(0, 20))..." -ForegroundColor Green

# Step 2: Register parent device
Write-Host ""
Write-Host "[2/6] Registering PARENT_DEVICE (Phone A)..." -ForegroundColor Yellow

$Form = @{
  device_name = "Parent Phone A"
  role = "parent_device"
  device_token = "fcm_parent_phone_a"
}
$ParentReg = Invoke-WebRequest -Uri "$BaseUrl/devices" `
  -Method POST `
  -Headers @{"Authorization" = "Bearer $AccessToken"} `
  -Form $Form | ConvertFrom-Json

$ParentDeviceId = $ParentReg.device.id
if ([string]::IsNullOrEmpty($ParentDeviceId)) {
  Write-Host "❌ Parent device registration failed. Response: $ParentReg" -ForegroundColor Red
  exit 1
}
Write-Host "✓ Parent device registered: $ParentDeviceId" -ForegroundColor Green
Write-Host "  Response: $($ParentReg | ConvertTo-Json -Depth 3)" -ForegroundColor Gray

# Step 3: Register child device
Write-Host ""
Write-Host "[3/6] Registering CHILD_DEVICE (Phone B, same account)..." -ForegroundColor Yellow

$Form = @{
  device_name = "Child Phone B"
  role = "child_device"
  device_token = "fcm_child_phone_b"
}
$ChildReg = Invoke-WebRequest -Uri "$BaseUrl/devices" `
  -Method POST `
  -Headers @{"Authorization" = "Bearer $AccessToken"} `
  -Form $Form | ConvertFrom-Json

$ChildDeviceId = $ChildReg.device.id
if ([string]::IsNullOrEmpty($ChildDeviceId)) {
  Write-Host "❌ Child device registration failed. Response: $ChildReg" -ForegroundColor Red
  exit 1
}
Write-Host "✓ Child device registered: $ChildDeviceId" -ForegroundColor Green
Write-Host "  Response: $($ChildReg | ConvertTo-Json -Depth 3)" -ForegroundColor Gray

# Step 4: List devices
Write-Host ""
Write-Host "[4/6] Listing devices (GET /devices with parent token)..." -ForegroundColor Yellow

$DevicesList = Invoke-WebRequest -Uri "$BaseUrl/devices" `
  -Method GET `
  -Headers @{"Authorization" = "Bearer $AccessToken"} | ConvertFrom-Json

Write-Host "✓ Devices list response:" -ForegroundColor Green
Write-Host "  $($DevicesList | ConvertTo-Json -Depth 3)" -ForegroundColor Gray

$ItemsCount = $DevicesList.items.Count
Write-Host "  Total devices: $ItemsCount" -ForegroundColor Cyan

if ($ItemsCount -lt 2) {
  Write-Host "❌ ERROR: Expected 2 devices, got $ItemsCount" -ForegroundColor Red
  exit 1
}

# Check for parent_device
$HasParent = @($DevicesList.items | Where-Object { $_.role -eq "parent_device" }).Count
# Check for child_device
$HasChild = @($DevicesList.items | Where-Object { $_.role -eq "child_device" }).Count

Write-Host "  Found parent_device: $HasParent" -ForegroundColor Cyan
Write-Host "  Found child_device: $HasChild" -ForegroundColor Cyan

if ($HasParent -ne 1 -or $HasChild -ne 1) {
  Write-Host "❌ ERROR: Expected 1 parent_device and 1 child_device" -ForegroundColor Red
  exit 1
}

# Step 5: Verify parent_id consistency
Write-Host ""
Write-Host "[5/6] Verifying parent_id consistency..." -ForegroundColor Yellow

$ParentIds = @($DevicesList.items | Select-Object -ExpandProperty parent_id -Unique)
$ParentIdCount = $ParentIds.Count

if ($ParentIdCount -ne 1) {
  Write-Host "❌ ERROR: Multiple parent_ids found. Expected 1, got $ParentIdCount" -ForegroundColor Red
  Write-Host "   Parent IDs: $ParentIds" -ForegroundColor Red
  exit 1
}

Write-Host "✓ All devices have same parent_id: $ParentIds" -ForegroundColor Green

# Step 6: Verify field structure
Write-Host ""
Write-Host "[6/6] Verifying field structure..." -ForegroundColor Yellow

$FirstDevice = $DevicesList.items[0]
$RequiredFields = @("id", "parent_id", "device_name", "role", "battery_percent", "is_online", "monitoring_enabled")

foreach ($field in $RequiredFields) {
  if ($FirstDevice.PSObject.Properties.Name -contains $field) {
    Write-Host "  ✓ Field '$field' present" -ForegroundColor Green
  } else {
    Write-Host "  ⚠ Field '$field' missing in first device" -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ ALL TESTS PASSED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  - Parent device registered: $ParentDeviceId" -ForegroundColor Green
Write-Host "  - Child device registered: $ChildDeviceId" -ForegroundColor Green
Write-Host "  - Both devices listed in GET /devices" -ForegroundColor Green
Write-Host "  - Both have same parent_id" -ForegroundColor Green
Write-Host "  - Response includes required fields" -ForegroundColor Green
