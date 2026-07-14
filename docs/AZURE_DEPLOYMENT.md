# Azure Student Deployment

This portfolio deployment uses Azure Container Apps for the FastAPI API and Dash dashboard. Both apps have one maximum replica and scale to zero when idle. The generated synthetic inputs and default experiment artifacts are built into the images, so the demo does not need a database or persistent storage.

The managed environment sends no application logs to Log Analytics. This is intentional for the low-cost demo; use the health endpoints and GitHub deployment history for initial verification.

## Cost Boundary

Use this design only for a low-traffic portfolio demo. It deliberately excludes virtual machines, Kubernetes, Azure ML, databases, virtual networking, custom domains, paid LLM calls, and Azure Container Registry.

Azure Container Apps includes a monthly free allowance of 180,000 vCPU-seconds, 360,000 GiB-seconds, and 2 million HTTP requests per subscription. Scale-to-zero avoids runtime usage while no replica is running. Check your Azure for Students balance before and after the first deployment. A budget notification is useful, but it does not stop resources.

## One-Time Azure Setup

Use **Azure Cloud Shell** in the Azure portal. It already includes the Azure CLI, so no local installation is needed. Azure for Students restricts deployment regions per subscription: first open **Policy > Assignments**, select **Allowed resource deployment regions**, and copy one value from its **Allowed locations** parameter. Use that exact location code below. The role assignment is restricted to this project resource group.

```powershell
$resourceGroup = 'rg-aircraft-maintenance-demo'
$location = '<allowed-location-code>'
$repository = 'lhnsdbc/Airline-Maintenance-Simulator-Public'

az provider register --namespace Microsoft.App
az provider show --namespace Microsoft.App --query registrationState --output tsv
az group create --name $resourceGroup --location $location

$subscriptionId = az account show --query id --output tsv
$tenantId = az account show --query tenantId --output tsv
$applicationId = az ad app create --display-name 'aircraft-maintenance-github-deploy' --query appId --output tsv
az ad sp create --id $applicationId

$scope = az group show --name $resourceGroup --query id --output tsv
az role assignment create --assignee $applicationId --role Contributor --scope $scope

$federatedCredential = @{
  name = 'github-main'
  issuer = 'https://token.actions.githubusercontent.com'
  subject = "repo:$($repository):ref:refs/heads/main"
  audiences = @('api://AzureADTokenExchange')
} | ConvertTo-Json -Compress

az ad app federated-credential create --id $applicationId --parameters $federatedCredential

Write-Output "AZURE_CLIENT_ID=$applicationId"
Write-Output "AZURE_TENANT_ID=$tenantId"
Write-Output "AZURE_SUBSCRIPTION_ID=$subscriptionId"
```

Wait until the provider query returns `Registered` before deploying. If `az ad app` commands are blocked by your university tenant, create an App registration in the Azure portal, add a federated credential for the `main` branch of this repository, and ask the subscription owner to assign it the Contributor role on `rg-aircraft-maintenance-demo`.

## GitHub Setup And First Deployment

1. In the GitHub repository, add three Actions secrets using the values Cloud Shell printed: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_SUBSCRIPTION_ID`.
2. Open **Actions > Deploy Azure Container Apps > Run workflow**. Keep `deploy` unchecked for the first run. This builds and publishes both container images.
3. In GitHub, open the two newly created packages and set their visibility to **Public**:
   - `aircraft-maintenance-simulator-api`
   - `aircraft-maintenance-simulator-dashboard`
4. Run the workflow again with `deploy` checked. It deploys the exact images built from that commit using OpenID Connect; no Azure password or client secret is stored in GitHub.
5. In the completed workflow, open the Azure CLI step and copy `apiUrl` and `dashboardUrl` from the deployment output.

## Verification

Run these requests in Cloud Shell after deployment, replacing the URLs with the outputs from the workflow:

```powershell
Invoke-RestMethod 'https://YOUR-API-URL/health'
Invoke-RestMethod 'https://YOUR-API-URL/experiments'
Invoke-RestMethod 'https://YOUR-DASHBOARD-URL/health'
```

The first request after inactivity can take time because the app starts from zero replicas. New experiment results written through the API are ephemeral and can disappear after scale-down; that is appropriate for this synthetic portfolio demo.

## Tear Down

To stop all possible charges, delete the resource group:

```powershell
az group delete --name rg-aircraft-maintenance-demo --yes
```
