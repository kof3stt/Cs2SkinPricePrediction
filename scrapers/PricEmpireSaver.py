import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


class JSONSaver:
    def __init__(self, filename: str = "items_data.json"):
        self.filename = filename
        self.data = self.load_existing_data()

    def load_existing_data(self) -> Dict[str, Any]:
        """Загружает существующие данные из файла, если он существует"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                return self._create_empty_structure()
        else:
            return self._create_empty_structure()

    def _create_empty_structure(self) -> Dict[str, Any]:
        """Создает пустую структуру данных"""
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "total_items": 0,
                "last_item_added": None,
            },
            "items": [],
        }

    def save_item(self, item_data: Dict[str, Any]):
        """Сохраняет данные о предмете"""
        self.data["metadata"]["updated"] = datetime.now().isoformat()
        self.data["metadata"]["last_item_added"] = datetime.now().isoformat()
        
        item_data["collected_at"] = datetime.now().isoformat()
        
        self.data["items"].append(item_data)
        self.data["metadata"]["total_items"] = len(self.data["items"])

        self.save_to_file()

    def save_items_batch(self, items_data: List[Dict[str, Any]]):
        """Сохраняет несколько предметов за один раз"""
        self.data["metadata"]["updated"] = datetime.now().isoformat()
        self.data["metadata"]["last_item_added"] = datetime.now().isoformat()
        
        for item_data in items_data:
            item_data["collected_at"] = datetime.now().isoformat()
            self.data["items"].append(item_data)
        
        self.data["metadata"]["total_items"] = len(self.data["items"])
        self.save_to_file()

    def save_to_file(self):
        """Сохраняет данные в файл"""
        os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else ".", exist_ok=True)
        
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)

    def clear_all_data(self):
        """Очищает все данные (сбрасывает к пустой структуре)"""
        self.data = self._create_empty_structure()
        self.save_to_file()

    def backup_data(self, backup_prefix: str = "backup"):
        """Создает резервную копию данных"""
        if not self.data["items"]:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_prefix}_{timestamp}_{os.path.basename(self.filename)}"
        
        with open(backup_filename, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)
        
        return backup_filename
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """Возвращает все сохраненные предметы"""
        return self.data["items"]