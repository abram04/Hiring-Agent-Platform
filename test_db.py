import asyncio
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

async def test_connection():
    conn = await asyncpg.connect(DATABASE_URL)
    
    # Test 1: Cek koneksi
    version = await conn.fetchval("SELECT version()")
    print(f"✅ Konek berhasil!")
    print(f"   PostgreSQL: {version[:50]}")
    
    # Test 2: Cek tabel ada
    tables = await conn.fetch("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    print(f"\n✅ Tabel ditemukan:")
    for t in tables:
        print(f"   - {t['table_name']}")
    
    # Test 3: Cek data
    jobs = await conn.fetchval("SELECT COUNT(*) FROM job_listings")
    candidates = await conn.fetchval("SELECT COUNT(*) FROM candidates")
    print(f"\n✅ Data:")
    print(f"   - job_listings : {jobs} rows")
    print(f"   - candidates   : {candidates} rows")
    
    await conn.close()

asyncio.run(test_connection())