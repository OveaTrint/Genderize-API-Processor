import os
from datetime import datetime, timezone

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()
GENDERIZE_API_KEY = os.getenv("GENDERIZE_API_KEY")

app = FastAPI(title="Genderize API Server")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# handles 422 exception
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request):
    return JSONResponse(
        {"status": "error", "message": "name parameter is not a string"},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


@app.get("/api/classify")
async def classify(name: str | None = None):
    try:
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing or empty name parameter",
            )

        async with httpx.AsyncClient() as client:
            url = "https://api.genderize.io"
            headers = {"Authorization": f"Bearer {GENDERIZE_API_KEY}"}
            params = {"name": name}

            response = await client.get(
                url=url,
                headers=headers,
                params=params,
                timeout=5,
            )

            # raises an error if anything goes wrong with calling the external api
            response.raise_for_status()

            data = response.json()
            # special edge case
            if data.get("gender") is None or data.get("count") == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No prediction available for provided name",
                )

            is_confident = (data.get("probability") >= 0.7) and (
                data.get("count") >= 100
            )
            processed_at = str(
                datetime.now(timezone.utc)
                .isoformat(
                    timespec="seconds",
                )
                .replace("+00:00", "Z")
            )

            return JSONResponse(
                content={
                    "status": "success",
                    "data": {
                        "name": name,
                        "gender": data.get("gender"),
                        "probability": data.get("probability"),
                        "sample_size": data.get("count"),
                        "is_confident": is_confident,
                        "processed_at": processed_at,
                    },
                }
            )
    # handle api errors
    except (httpx.HTTPStatusError, httpx.RequestError):
        return JSONResponse(
            content={"status": "error", "message": "Bad Gateway"},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    # handle custom raised errors
    except HTTPException as exc:
        return JSONResponse(
            content={"status": "error", "message": exc.detail},
            status_code=exc.status_code,
        )
    # handle unexpected errors
    except Exception:
        return JSONResponse(
            content={"status": "error", "message": "Something went wrong"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


if __name__ == "__main__":
    uvicorn.run(app)
