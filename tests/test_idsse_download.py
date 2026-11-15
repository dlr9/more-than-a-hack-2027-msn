"""Synthetic HTTP streams only; no network, licensed assets or real matches."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from scripts.download_idsse import download_http, verify_xml, safe_error, acquire, filename

XML=b'<?xml version="1.0"?><Positions><FrameSet><Frame N="0"/></FrameSet></Positions>'

class Response(io.BytesIO):
    def __init__(self, body, status=200, content_range=None):
        super().__init__(body);self.status=status
        self.headers={'Content-Range':content_range} if content_range else {}

class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.dest=Path(self.tmp.name)/'position.xml'
        self.kw=dict(size=len(XML),md5=hashlib.md5(XML).hexdigest(),kind='positions')

    def test_resume_206_aligned(self):
        self.dest.with_suffix('.xml.part').write_bytes(XML[:19])
        opener=Mock(return_value=Response(XML[19:],206,f'bytes 19-{len(XML)-1}/{len(XML)}'))
        result=download_http('https://example.invalid/file',self.dest,opener=opener,**self.kw)
        self.assertEqual(opener.call_args.args[0].get_header('Range'),'bytes=19-')
        self.assertEqual(self.dest.read_bytes(),XML);self.assertEqual(result['http_status'],206)

    def test_ignored_range_200_replaces_partial(self):
        self.dest.with_suffix('.xml.part').write_bytes(b'broken-prefix')
        download_http('https://example.invalid/file',self.dest,opener=Mock(return_value=Response(XML)),**self.kw)
        self.assertEqual(self.dest.read_bytes(),XML)

    def test_misaligned_range_preserves_partial(self):
        partial=self.dest.with_suffix('.xml.part');partial.write_bytes(XML[:19])
        with self.assertRaisesRegex(ValueError,'Content-Range'):
            download_http('https://example.invalid/file',self.dest,opener=Mock(return_value=Response(XML[19:],206,f'bytes 18-{len(XML)-1}/{len(XML)}')),**self.kw)
        self.assertEqual(partial.read_bytes(),XML[:19]);self.assertFalse(self.dest.exists())

    def test_corrupt_hash_never_promoted(self):
        bad=XML.replace(b'N="0"',b'N="1"')
        with self.assertRaisesRegex(ValueError,'md5 mismatch'):
            download_http('https://example.invalid/file',self.dest,opener=Mock(return_value=Response(bad)),**self.kw)
        self.assertFalse(self.dest.exists())

    def test_existing_verified_cache_never_requests_network(self):
        self.dest.write_bytes(XML);opener=Mock(side_effect=AssertionError('network called'))
        download_http('https://example.invalid/file',self.dest,opener=opener,**self.kw)
        opener.assert_not_called()

    def test_existing_corrupt_cache_is_preserved_not_overwritten(self):
        self.dest.write_bytes(b'broken');opener=Mock()
        with self.assertRaises(ValueError):download_http('https://example.invalid/file',self.dest,opener=opener,**self.kw)
        self.assertEqual(self.dest.read_bytes(),b'broken');opener.assert_not_called()

    def test_lfs_html_and_wrong_xml_type_rejected(self):
        for payload in (b'version https://git-lfs.github.com/spec/v1',b'<html>forbidden</html>',b'<Other/>'):
            self.dest.write_bytes(payload)
            with self.assertRaises(ValueError):verify_xml(self.dest,len(payload),kind='positions')

    def test_complete_partial_validates_without_network(self):
        self.dest.with_suffix('.xml.part').write_bytes(XML);opener=Mock()
        download_http('https://example.invalid/file',self.dest,opener=opener,**self.kw)
        opener.assert_not_called();self.assertTrue(self.dest.exists())

    def test_full_acquisition_cache_is_offline(self):
        directory=Path(self.tmp.name)
        (directory/filename('J03WPY','positions')).write_bytes(XML)
        manifest={'J03WPY':{'positions':(1,len(XML),hashlib.md5(XML).hexdigest(),hashlib.sha256(XML).hexdigest())}}
        with patch('scripts.download_idsse.MANIFEST',manifest), patch('scripts.download_idsse.figshare_metadata') as metadata:
            acquire('J03WPY',directory,directory/'attempts.jsonl')
        metadata.assert_not_called()

    def test_figshare_failure_uses_verified_hf_stage(self):
        directory=Path(self.tmp.name)
        stage=directory/'.hf-stage';stage.mkdir()
        (stage/filename('J03WPY','positions')).write_bytes(XML)
        manifest={'J03WPY':{'positions':(1,len(XML),hashlib.md5(XML).hexdigest(),hashlib.sha256(XML).hexdigest())}}
        with patch('scripts.download_idsse.MANIFEST',manifest), patch('scripts.download_idsse.figshare_metadata',side_effect=RuntimeError('403')), patch('scripts.download_idsse.download_http',side_effect=RuntimeError('403')), patch('scripts.download_idsse.subprocess.run',return_value=Mock(returncode=0)) as hf:
            acquire('J03WPY',directory,directory/'attempts.jsonl')
        self.assertEqual((directory/filename('J03WPY','positions')).read_bytes(),XML)
        self.assertEqual(hf.call_count,1)
        self.assertIn('huggingface',(directory/'attempts.jsonl').read_text())

    def test_reachable_metadata_conflict_does_not_silently_fallback(self):
        with patch('scripts.download_idsse.figshare_metadata',side_effect=ValueError('unexpected license')), patch('scripts.download_idsse.subprocess.run') as hf:
            with self.assertRaisesRegex(ValueError,'license'):
                acquire('J03WPY',Path(self.tmp.name),Path(self.tmp.name)/'attempts.jsonl')
        hf.assert_not_called()

    def test_error_strips_signed_queries(self):
        message=safe_error(RuntimeError('403 https://cdn.example.test/x?X-Amz-Signature=secret&token=key'))
        self.assertNotIn('secret',message);self.assertNotIn('token',message)
        self.assertIn('https://cdn.example.test/x',message)

if __name__ == '__main__': unittest.main()
