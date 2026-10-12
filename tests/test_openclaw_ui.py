import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from integrations.openclaw.prepare_ui import prepare

ROOT = Path(__file__).resolve().parents[1]


class OpenClawPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root/'openclaw'
        self.ui = self.package/'dist/control-ui'
        self.ui.mkdir(parents=True)
        (self.package/'package.json').write_text(json.dumps({'name': 'openclaw', 'version': '2026.3.13'}))
        (self.ui/'index.html').write_text('<html><head><script src="./agent-workspace.js"></script>'
                                        '<script type="module" src="./app.js"></script></head>'
                                        '<body><openclaw-app></openclaw-app></body></html>')
        (self.ui/'app.js').write_text('// vendor fixture')
        self.out = self.root/'prepared'

    def test_portable_ui_preserves_live_files_and_private_config(self):
        original = (self.ui/'index.html').read_bytes()
        base = {'gateway': {'auth': {'token': 'LOCAL_TEST_TOKEN'}},
                'agents': {'list': [{'id': 'main', 'workspace': '/existing/main'}]}}
        prepare(self.package, self.out, base_config=base)
        html = (self.out/'control-ui/index.html').read_text()
        self.assertEqual(html.count('agent-workspace.js'), 1)
        self.assertLess(html.index('dentalclaw-config.js'), html.index('workflow.js'))
        self.assertLess(html.index('agent-workspace.js'), html.index('type="module"'))
        self.assertEqual((self.ui/'index.html').read_bytes(), original)
        self.assertEqual(len(base['agents']['list']), 1)
        merged = json.loads((self.out/'openclaw.local.json').read_text())
        self.assertEqual(merged['gateway']['auth']['token'], 'LOCAL_TEST_TOKEN')
        self.assertEqual([a['id'] for a in merged['agents']['list']], ['main', 'dentalclaw'])
        self.assertEqual((self.out/'openclaw.local.json').stat().st_mode & 0o777, 0o600)
        public = ''.join(p.read_text() for p in (self.out/'control-ui').iterdir() if p.is_file())
        self.assertNotIn('LOCAL_TEST_TOKEN', public)
        with self.assertRaises(ValueError):
            prepare(self.package, self.out)

    def test_context_contents_are_not_published(self):
        context = self.root/'context.json'
        context.write_text(json.dumps({'data_root': '/private/source-not-for-the-browser'}))
        prepare(self.package, self.out, profiles=[{'id': 'qc', 'label': 'Package audit', 'contextPath': str(context)}])
        public = (self.out/'control-ui/dentalclaw-config.js').read_text()
        self.assertNotIn('/private/source-not-for-the-browser', public)
        self.assertIn(str(context), public)

    def test_existing_agent_and_reference_label_context_rejected_before_writes(self):
        with self.assertRaises(ValueError):
            prepare(self.package, self.out, base_config={'agents': {'list': [{'id': 'dentalclaw'}]}})
        context = self.root/'context.json'
        context.write_text('{"expected_action":"route_ready"}')
        with self.assertRaises(ValueError):
            prepare(self.package, self.out, profiles=[{'id': 'bad', 'label': 'Bad', 'contextPath': str(context)}])
        self.assertFalse(self.out.exists())

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser contract tests')
    def test_javascript_execution_controls(self):
        subprocess.run(['node', '--test', str(ROOT/'tests/test_openclaw_workflow.cjs')],
                       check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser contract tests')
    def test_browser_command_reaches_four_route_cli_without_network(self):
        repo = self.root/'portable repo'
        repo.mkdir()
        (repo/'platform_mvp').symlink_to(ROOT/'platform_mvp', target_is_directory=True)
        context = self.root/'context.json'
        context.write_text('{}')
        decision = self.root/'decision.json'
        decision.write_text(json.dumps({'decision': 'clarify', 'route_id': None,
            'clarification_fields': ['task'], 'confidence': 0.5, 'reasoning': 'Offline fixture',
            'parameters': {'case_ids': [], 'artifact_kinds': [], 'table': False, 'selection_required': False}}))
        config = {'schemaVersion': 1, 'repoRoot': str(repo), 'python': sys.executable,
                  'profiles': [{'id': 'input', 'label': 'Input', 'contextPath': str(context)}]}
        script = "const w=require(process.argv[1]);console.log(JSON.stringify(w.build(JSON.parse(process.argv[2]),'Clarify my research request',{profileId:'input',runId:'offline_bridge'})));"
        result = subprocess.run(['node', '-e', script, str(ROOT/'integrations/openclaw/web/workflow.js'), json.dumps(config)],
                                check=True, capture_output=True, text=True)
        command = json.loads(result.stdout)
        subprocess.run(command['argv'] + ['--decision-json', str(decision)], check=True, capture_output=True, text=True)
        prediction = json.loads((Path(command['runDir'])/'prediction.json').read_text())
        self.assertEqual(prediction['action'], 'clarify')
        self.assertFalse(prediction['execution_permission'])
        self.assertFalse((Path(command['runDir'])/'execution_result.json').exists())

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser contract tests')
    def test_browser_contract_and_shell_argument_roundtrip(self):
        module = ROOT/'integrations/openclaw/web/workflow.js'
        requirement = "A very long request " * 30 + "\n中文 '$HOME' `touch /tmp/dentalclaw-should-not-exist` $(false)"
        config = {'schemaVersion': 1, 'repoRoot': str(ROOT), 'python': sys.executable,
                  'profiles': [{'id': 'input', 'label': 'Input', 'contextPath': '/a path/context.json'}]}
        script = "const w=require(process.argv[1]);const c=JSON.parse(process.argv[2]);const r=process.argv[3];console.log(JSON.stringify(w.build(c,r,{profileId:'input',runId:'offline_test'})));"
        result = subprocess.run(['node', '-e', script, str(module), json.dumps(config), requirement],
                                check=True, capture_output=True, text=True)
        workflow = json.loads(result.stdout)
        self.assertEqual(workflow['argv'][workflow['argv'].index('--intent') + 1], requirement)
        self.assertIn('--next-version', workflow['argv'])
        self.assertNotIn('--execute', workflow['argv'])
        self.assertNotIn('--case-id', workflow['argv'])
        self.assertNotIn('--reuse-inference', workflow['argv'])
        # Parse through a real shell but replace the executable with printf: no model/API invocation.
        from shlex import split
        self.assertEqual(split(workflow['command']), workflow['argv'])
        out = subprocess.run(['sh', '-c', "set -- " + workflow['command'] + "; printf '%s\\0' \"$@\""],
                             check=True, capture_output=True)
        self.assertEqual(out.stdout.decode().split('\0')[:-1], workflow['argv'])


if __name__ == '__main__':
    unittest.main()
