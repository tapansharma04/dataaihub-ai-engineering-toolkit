# Model catalog/history transactions

Supported provider updates coordinate `catalog.json` and
`lifecycle-history.json` through a local write-ahead transaction. The legacy
files remain the committed representation and remain readable by existing
installations.

One writer holds `models/.catalog-history.lock` while it recovers any pending
transaction, reads the current pair, calculates the provider update, and
commits it. Lock acquisition waits for a bounded five seconds. A writer which
waits acquires the lock only after the preceding writer has finished or its
process has released the OS lock.

The transaction first validates and durably stages both after-images, then
writes an intent journal containing before and after hashes. It atomically
replaces catalog and history from those stages, updating the journal after
each visible replacement. The final durable `committed` journal state is the
commit acknowledgement; cleanup is best-effort and recoverable.

On the next transaction-aware read or write, recovery validates the live file
hashes against the journal. A matching incomplete state is completed from the
staged after-images. A missing stage, corrupt journal, or a file changed
outside the transaction fails closed for manual recovery. Recovery never
contacts provider documentation.

`FileCatalogStore.load()` and `FileLifecycleHistoryStore.load()` recover a
pending supported transaction before loading. Callers that need a pair should
use `load_catalog_history()` to hold the coordination lock while reading both.
Programs that open the two legacy files directly cannot be promised an atomic
cross-file snapshot during the narrow pair-replacement interval.

The protocol relies on atomic same-directory replacement and file/directory
sync where supported. It cannot make stronger guarantees on filesystems that
do not honor those primitives or where directory synchronization is ignored.
