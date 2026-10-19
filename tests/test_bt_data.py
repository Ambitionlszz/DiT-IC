import hashlib
from pathlib import Path
import tempfile
import unittest

from models.bt_self_flow.data import EXTENSIONS, ImageDataset


class DataInventoryTest(unittest.TestCase):
    def test_inventory_preserves_order_fingerprint_and_symlink_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            root.mkdir()
            for name in ("z.JPG", "nested/a.PNG", ".hidden/c.webp", "dir.png/b.bmp", "notes.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            outside = Path(directory) / "outside"
            outside.mkdir()
            (outside / "excluded.jpg").touch()
            (root / "linked.jpg").symlink_to(root / "z.JPG")
            (root / "directory_link").symlink_to(outside, target_is_directory=True)
            (root / "broken.png").symlink_to(root / "missing.png")
            roots = [root, root / "nested"]
            expected = sorted({path for base in roots for path in base.rglob("*")
                               if path.suffix.lower() in EXTENSIONS and path.is_file()})
            self.assertEqual(len(expected), 5)
            for limit in (None, 3):
                dataset = ImageDataset(roots, limit=limit)
                paths = expected[:limit]
                self.assertEqual(dataset.paths, paths)
                fingerprint = hashlib.sha256("\n".join(map(str, paths)).encode()).hexdigest()
                self.assertEqual(dataset.fingerprint, fingerprint)

    def test_missing_and_empty_directories_remain_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing image directory"):
                ImageDataset([Path(directory) / "missing"])
            with self.assertRaisesRegex(ValueError, "image dataset is empty"):
                ImageDataset([directory])


if __name__ == "__main__":
    unittest.main()
