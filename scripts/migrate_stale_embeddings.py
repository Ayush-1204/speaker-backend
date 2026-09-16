import argparse, os, shutil, sys, glob, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('migrate')
EXPECTED_DIM = 192
DATA_ROOT = (os.environ.get('SAFEEAR_DATA_ROOT', os.path.join('app','data')) or '').strip() or os.path.join('app','data')
TENANTS_ROOT = os.path.join(DATA_ROOT, 'tenants')
QUARANTINE_DIR = os.path.join(DATA_ROOT, 'stale_dim_quarantine')

def _infer_dim(path):
    sz = os.path.getsize(path); p = sz - 128
    return p // 4 if p > 0 and p % 4 == 0 else -1

def _load_dim(path):
    try:
        import numpy as np
        return int(np.load(path, allow_pickle=False).reshape(-1).shape[0])
    except Exception: return -1

def _get_db():
    try:
        from app.database.session import SessionLocal
        return SessionLocal()
    except Exception as e:
        logger.error('Cannot open DB: %s', e); return None

def _update_db(db, parent_id, speaker_id, valid_count):
    try:
        from app.database.models import EnrolledSpeaker
        sp = db.query(EnrolledSpeaker).filter(
            EnrolledSpeaker.parent_id == parent_id,
            EnrolledSpeaker.id == speaker_id).first()
        if sp is None:
            logger.warning('NOT_FOUND parent=%s speaker=%s', parent_id, speaker_id)
            return False
        sp.sample_count = valid_count
        if hasattr(sp, 'quality_label'):
            sp.quality_label = 'needs_reenrollment' if valid_count == 0 else 'poor'
        db.commit()
        return True
    except Exception as e:
        logger.error('DB_UPDATE_FAILED %s', e)
        try: db.rollback()
        except Exception: pass
        return False

def scan(dry_run):
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    db = None if dry_run else _get_db()
    report, total, stale_n = [], 0, 0
    if not os.path.isdir(TENANTS_ROOT):
        logger.error('TENANTS_ROOT not found: %s', TENANTS_ROOT)
        sys.exit(1)
    for parent_id in sorted(os.listdir(TENANTS_ROOT)):
        emb_root = os.path.join(TENANTS_ROOT, parent_id, 'embeddings')
        if not os.path.isdir(emb_root): continue
        for speaker_id in sorted(os.listdir(emb_root)):
            sdir = os.path.join(emb_root, speaker_id)
            if not os.path.isdir(sdir): continue
            files = sorted(glob.glob(os.path.join(sdir, '*.npy')))
            good, bad = [], []
            for f in files:
                total += 1
                d = _infer_dim(f)
                if d == EXPECTED_DIM: good.append(f); continue
                real = _load_dim(f)
                if real == EXPECTED_DIM: good.append(f)
                else: bad.append((f, real if real != -1 else d))
            if not bad: continue
            stale_n += len(bad)
            for fpath, dim in bad:
                fname = os.path.basename(fpath)
                qpath = os.path.join(QUARANTINE_DIR, parent_id, speaker_id, fname)
                logger.info('%s parent=%s speaker=%s file=%s dim=%s',
                    '[DRY-RUN]' if dry_run else 'QUARANTINE',
                    parent_id[:8], speaker_id[:8], fname, dim)
                if not dry_run:
                    os.makedirs(os.path.dirname(qpath), exist_ok=True)
                    shutil.move(fpath, qpath)
            db_ok = _update_db(db, parent_id, speaker_id, len(good)) if (db and not dry_run) else False
            report.append({'parent_id': parent_id, 'speaker_id': speaker_id,
                'good': len(good),
                'stale': [(os.path.basename(f), d) for f, d in bad],
                'db_ok': db_ok})
    print()
    print('=' * 72)
    print('STALE EMBEDDING MIGRATION REPORT')
    print('  Mode    :', 'DRY RUN' if dry_run else 'LIVE')
    print('  Total   :', total, '.npy files scanned')
    print('  Stale   :', stale_n, 'files with wrong dimension')
    print('  Speakers:', len(report), 'affected')
    print('=' * 72)
    if not report:
        print('No stale embeddings found. Nothing to do.')
    else:
        hdr = '{:<24} {:<24} {:>5} {:>6}  {}'
        print(hdr.format('Parent', 'Speaker', 'Good', 'Stale', 'DB'))
        print('-' * 70)
        for e in sorted(report, key=lambda x: x['parent_id']):
            items = e['stale']
            stale_str = ', '.join(str(items[i][0]) + '(' + str(items[i][1]) + 'd)' for i in range(len(items)))
            print(hdr.format(e['parent_id'][:22], e['speaker_id'][:22],
                e['good'], len(e['stale']),
                'ok' if e['db_ok'] else 'skipped'))
            print('  stale:', stale_str)
        print()
        if dry_run: print('Re-run without --dry-run to apply.')
        else:
            print('Quarantine:', QUARANTINE_DIR)
            print('Affected speakers set to quality_label=needs_reenrollment in DB.')
    if db:
        try: db.close()
        except Exception: pass

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Quarantine stale-dimension embeddings')
    p.add_argument('--dry-run', action='store_true')
    scan(dry_run=p.parse_args().dry_run)

