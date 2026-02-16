from config import settings

def main() -> None:
    print("✅ Project booted")
    print(f"Project root: {settings.project_root}")
    print(f"Data dir: {settings.data_dir}")

if __name__ == "__main__":
    main()
