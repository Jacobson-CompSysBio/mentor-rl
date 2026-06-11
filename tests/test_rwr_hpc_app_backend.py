import unittest
import tempfile
from pathlib import Path

from runtime.rwr_hpc_app_backend import RwrHpcAppBackend

class RwrHpcAppBackendTests(unittest.TestCase):
    # create a temp directory with a fake "rwr_loe" executable for testing
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.tmp.name)

        # set up a fake manifest that lists the rwr_loe executable
        app_dir = self.build_dir / "apps" / "rwr_loe"
        app_dir.mkdir(parents=True)

        # create a fake rwr_loe executable that just prints its arguments
        self.rwr_loe = app_dir / "rwr_loe"
        self.rwr_loe.write_text(
            "#!/usr/bin/env bash\n"
            "echo 'fake rwr_loe output'\n",
            encoding="utf-8",
        )
        # make it executable
        self.rwr_loe.chmod(0o755)
    
    def tearDown(self) -> None:
        self.tmp.cleanup()
    
    # test that rwr_loe executable is discoverable in the build dir
    def test_discoverable_loe_executable(self) -> None:
        backend = RwrHpcAppBackend(build_dir=self.build_dir)

        self.assertIn("rwr_loe", backend.apps)
        self.assertTrue(backend.apps["rwr_loe"].exists())

    # report missing required required apps
    def test_missing_required_apps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = RwrHpcAppBackend(build_dir=tmp)
        
        self.assertIn("rwr_loe", backend.missing_apps())

    # test that stdout, stderr, and return are captured
    def test_run_app_captures_output(self) -> None:
        backend = RwrHpcAppBackend(build_dir=self.build_dir)

        result = backend.run_app("rwr_loe", ["--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("fake rwr_loe output", result.stdout)
        self.assertEqual(result.stderr, "")

    # test that require_app("missing") raises a useful error msg
    def test_require_app_missing_raises(self) -> None:
        backend = RwrHpcAppBackend(build_dir=self.build_dir)
        with self.assertRaisesRegex(KeyError, "RWR\\+\\+ app 'missing' not found"):
            backend.require_app("missing")
    
    # test that executable path is correctly found from manifest
    def test_require_app_finds_executable(self) -> None:
        backend = RwrHpcAppBackend(
            build_dir=self.build_dir)
        executable = backend.require_app("rwr_loe")

        self.assertEqual(executable.name, "rwr_loe")
        self.assertTrue(executable.exists())
        self.assertTrue(executable.samefile(self.rwr_loe))
    
    def test_manifest_path_discovers_executable(self) -> None:
        manifest_path = self.build_dir / "rwr_hpc_apps.txt"
        manifest_path.write_text(f"{self.rwr_loe}\n", encoding="utf-8")

        backend = RwrHpcAppBackend(
            build_dir=self.build_dir,
            manifest_path=manifest_path,
        )

        executable = backend.require_app("rwr_loe")

        self.assertTrue(executable.samefile(self.rwr_loe))
        self.assertEqual(str(manifest_path.resolve()), str(backend.manifest_path))

if __name__ == "__main__":
    unittest.main()