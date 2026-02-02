#!/bin/bash
# =============================================================================
# Google Cloud Run Deployment Script
# Portfolio Chatbot Backend
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-chatbot-api}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
TIMEOUT="${TIMEOUT:-300}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Portfolio Chatbot - Cloud Run Deployment ===${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}Not logged in to gcloud. Running 'gcloud auth login'...${NC}"
    gcloud auth login
fi

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$CURRENT_PROJECT" ]; then
    echo -e "${YELLOW}No project set. Setting project to: ${PROJECT_ID}${NC}"
    gcloud config set project $PROJECT_ID
else
    PROJECT_ID=$CURRENT_PROJECT
    echo -e "${GREEN}Using project: ${PROJECT_ID}${NC}"
fi

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${YELLOW}Warning: GROQ_API_KEY environment variable is not set.${NC}"
    echo "You can set it with: export GROQ_API_KEY='your-api-key'"
    read -p "Enter your Groq API key (or press Enter to skip): " GROQ_API_KEY
fi

# Vertex AI project (backup LLM) - default to current project
if [ -z "$VERTEX_AI_PROJECT" ]; then
    VERTEX_AI_PROJECT=$PROJECT_ID
    echo -e "${YELLOW}Using current project for Vertex AI backup: ${VERTEX_AI_PROJECT}${NC}"
fi

echo ""
echo -e "${GREEN}Deployment Configuration:${NC}"
echo "  Project ID:    $PROJECT_ID"
echo "  Region:        $REGION"
echo "  Service Name:  $SERVICE_NAME"
echo "  Memory:        $MEMORY"
echo "  CPU:           $CPU"
echo "  Timeout:       ${TIMEOUT}s"
echo "  Min Instances: $MIN_INSTANCES"
echo "  Max Instances: $MAX_INSTANCES"
echo ""

read -p "Continue with deployment? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Step 1: Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    --quiet

echo ""
echo -e "${GREEN}Step 2: Building and deploying to Cloud Run...${NC}"

# Build deploy command
# --cpu-boost helps with cold start (faster model loading)
DEPLOY_CMD="gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --allow-unauthenticated \
    --memory $MEMORY \
    --cpu $CPU \
    --cpu-boost \
    --timeout $TIMEOUT \
    --max-instances $MAX_INSTANCES \
    --min-instances $MIN_INSTANCES \
    --port 8080"

# Add environment variables
if [ -n "$GROQ_API_KEY" ]; then
    DEPLOY_CMD="$DEPLOY_CMD --set-env-vars GROQ_API_KEY=$GROQ_API_KEY"
fi

# Add Vertex AI project for backup LLM
DEPLOY_CMD="$DEPLOY_CMD --set-env-vars VERTEX_AI_PROJECT=$VERTEX_AI_PROJECT"

# Add additional env vars if needed
DEPLOY_CMD="$DEPLOY_CMD --set-env-vars ENVIRONMENT=production"

# Execute deployment
eval $DEPLOY_CMD

echo ""
echo -e "${GREEN}Step 3: Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test your deployment:"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/docs"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Update your frontend with the new API URL"
echo "2. Test the chat endpoint: curl -X POST $SERVICE_URL/chat -H 'Content-Type: application/json' -d '{\"message\":\"Hello\"}'"
echo "3. Monitor logs: gcloud run services logs tail $SERVICE_NAME --region $REGION"
