"""
JWT Token 生成工具 - 前端使用
"""
import jwt
import datetime
import uuid

# JWT 密钥（与后端保持一致）- 必须 32 字节以上
SECRET_KEY = "HeartEchoSystem_JWT_Secret_Key_2024_32Bytes_Long"


def generate_token(username: str, expire_hours: int = 12) -> str:
    """
    生成 JWT Token
    
    Args:
        username: 用户名
        expire_hours: 过期时间（小时）
    
    Returns:
        JWT Token 字符串
    """
    payload = {
        "user_id": str(uuid.uuid4()),
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=expire_hours),
        "iat": datetime.datetime.utcnow(),  # 签发时间
        "type": "access"
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def verify_token(token: str) -> dict:
    """
    验证 JWT Token（前端本地验证用）
    
    Args:
        token: JWT Token
    
    Returns:
        payload 字典，验证失败返回 None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


if __name__ == "__main__":
    # 测试
    token = generate_token("admin")
    print(f"Generated Token: {token}")
    print(f"Verify Result: {verify_token(token)}")
