"""Transient object storage (MinIO / S3) helpers.

The bucket is a **staging area**, not a store: the plan (section 11) requires
zero retention of client images, so the workers delete the objects as soon as
the job reaches a terminal state and the bucket keeps a 1-day lifecycle rule as
a safety net (see ``kubernetes/manifests/README.md``).
"""
from __future__ import annotations

import os
from typing import Iterable

from . import DEFAULT_BUCKET


def endpoint_url() -> str | None:
    """S3 endpoint. ``MINIO_ENDPOINT`` is a host:port (in-cluster Service name)."""
    raw = os.environ.get("MINIO_ENDPOINT") or os.environ.get("S3_ENDPOINT")
    if not raw:
        return None
    return raw if raw.startswith("http") else f"http://{raw}"


def bucket_name() -> str:
    return os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET)


def s3_client():
    """Build a path-style S3v4 client (required by MinIO with DNS bucket names)."""
    import boto3  # lazy: only the cluster processes need it
    from botocore.config import Config

    user = os.environ.get("MINIO_ROOT_USER") or os.environ.get("AWS_ACCESS_KEY_ID")
    password = os.environ.get("MINIO_ROOT_PASSWORD") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url(),
        aws_access_key_id=user,
        aws_secret_access_key=password,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket(client, bucket: str | None = None) -> str:
    """Create the bucket if missing; returns the bucket name."""
    name = bucket or bucket_name()
    try:
        client.head_bucket(Bucket=name)
    except Exception:
        client.create_bucket(Bucket=name)
    return name


def put_bytes(client, key: str, data: bytes, content_type: str = "application/octet-stream", bucket: str | None = None) -> str:
    name = bucket or bucket_name()
    client.put_object(Bucket=name, Key=key, Body=data, ContentType=content_type)
    return key


def get_bytes(client, key: str, bucket: str | None = None) -> bytes:
    name = bucket or bucket_name()
    return client.get_object(Bucket=name, Key=key)["Body"].read()


def delete_objects(client, keys: Iterable[str], bucket: str | None = None) -> int:
    name = bucket or bucket_name()
    payload = [{"Key": key} for key in keys]
    if not payload:
        return 0
    client.delete_objects(Bucket=name, Delete={"Objects": payload, "Quiet": True})
    return len(payload)


def delete_prefix(client, prefix: str, bucket: str | None = None) -> int:
    """Delete every object under ``prefix`` (``<job_id>/``); returns the count."""
    name = bucket or bucket_name()
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in page.get("Contents", []))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return delete_objects(client, keys, bucket=name)


def presign_get(client, key: str, expires_seconds: int = 300, bucket: str | None = None) -> str:
    name = bucket or bucket_name()
    return client.generate_presigned_url("get_object", Params={"Bucket": name, "Key": key}, ExpiresIn=int(expires_seconds))
