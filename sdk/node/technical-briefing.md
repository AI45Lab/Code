# Technical Briefing

---

## 1. TCP Congestion Control (Slow Start / AIMD)

TCP congestion control prevents senders from overwhelming the network. It maintains a **congestion window** (`cwnd`) — the maximum number of unacknowledged bytes a sender may transmit — alongside the receiver's advertised window, and uses the effective minimum of the two.

**Slow Start** begins when a connection opens or after a loss event. `cwnd` starts at a small value (historically 1 MSS, now typically ~10 MSS per RFC 6928) and doubles every round-trip time: each arriving ACK lets the sender increase `cwnd` by 1 MSS. This exponential growth quickly probes available bandwidth but risks overshooting. Growth continues until `cwnd` reaches the **slow-start threshold** (`ssthresh`), at which point the algorithm transitions to the **AIMD** (Additive Increase / Multiplicative Decrease) phase.

In **AIMD**, `cwnd` grows linearly — increasing by roughly 1 MSS per RTT (additive increase) — allowing gentle bandwidth probing. When a loss is detected (triple duplicate ACKs), `cwnd` is cut in half (multiplicative decrease) and `ssthresh` is set to the reduced `cwnd`. This yields the classic "sawtooth" pattern: steady linear ramps punctuated by halving drops. A timeout (indicating severe congestion) resets `cwnd` to its initial value and triggers slow start again. The AIMD dynamic ensures convergence: competing flows sharing a bottleneck naturally reach a fair equilibrium, since a flow with a larger `cwnd` yields more absolute bytes on each multiplicative cut, gradually equalizing throughput across sessions.

---

## 2. B-Tree Database Index

A B-tree is a self-balancing ordered tree where every node stores sorted keys and child pointers. Each internal node holds up to *m*−1 keys and *m* pointers (its branching factor or fan-out), while leaf nodes store keys paired with record pointers to actual rows. A critical invariant guarantees that all leaf nodes reside at the same depth, so every lookup traverses the same number of levels — ensuring O(log *n*) worst-case search time.

**Search** binary-searches the keys within a node, follows the appropriate child pointer, and repeats until reaching a leaf. **Insert** descends to the target leaf, adds the key in sorted order, and — if the node overflows — splits it at its median: the median key propagates upward into the parent, and the node divides into two half-full siblings. Splits can cascade upward; if the root splits, a new root is created, increasing tree height by one.

Databases favor B-trees because their high fan-out (often hundreds of keys per node) keeps trees extremely shallow — typically 3–4 levels for millions of records. Since each node maps naturally to one disk page, a lookup requires only that many page reads, minimizing expensive disk I/O. This locality and predictability make B-trees the default index structure in virtually every relational database engine.

---

## 3. TLS 1.3 Handshake

The TLS 1.3 handshake establishes a secure channel in just **1 RTT** (round-trip time), a major reduction from TLS 1.2's 2-RTT handshake. The client sends a **ClientHello** that includes supported AEAD cipher suites and a `key_share` extension carrying an ephemeral Diffie-Hellman public key (typically X25519 or P-256), removing the need for a separate key-exchange round. The server responds with a **ServerHello** selecting the cipher suite and providing its own DH public key, allowing both sides to compute the shared secret immediately.

Using HKDF-based key derivation, the server then sends its Certificate, CertificateVerify, and Finished messages in a single encrypted flight. The client verifies the certificate, sends its own Finished, and application data can flow immediately after — all within one round trip. TLS 1.3 also supports **0-RTT resumption** via pre-shared keys (PSK), trading forward secrecy for near-instant reconnection on early data.

Key differences from TLS 1.2: RSA key exchange is eliminated entirely (enforcing forward secrecy by default), non-AEAD cipher suites and compression are removed, renegotiation is forbidden, and static RSA/ECDSA keys can no longer be used for key agreement. These changes dramatically reduce the attack surface and simplify the protocol.

---

## 4. Bloom Filter

A Bloom filter is a space-efficient probabilistic data structure for approximate set membership testing. It consists of a bit array of *m* bits (all initially 0) and *k* independent hash functions, each mapping an element to a position in the array. To **insert** an element, all *k* hashes are computed and the corresponding bits are set to 1. To **query** membership, the same *k* positions are checked: if every bit is 1, the element is *possibly* in the set; if any bit is 0, it is *definitely not* in the set. This one-sided error means false negatives are impossible, but false positives can occur when bits set by different elements coincidentally overlap.

The false-positive probability after inserting *n* elements is approximately **(1 − e^(−kn/m))^k** — the chance that a specific bit remains 0 after one hash is (1 − 1/m)^(kn) ≈ e^(−kn/m), so all *k* bits being 1 yields the formula above. For a given ratio *m/n*, the optimal number of hashes that minimizes this probability is **k ≈ (m/n) ln 2**, at which point the false-positive rate is roughly (½)^(m/n)·ln2 ≈ (0.6185)^(m/n). This allows engineers to size the filter precisely: e.g., 10 bits per element yields ~1% false positives with ~7 hash functions, far less space than a full hash table.

---

## 5. Consistent Hashing

Consistent hashing maps objects and nodes onto a circular hash space (0…2¹⁶⁰−1) called a "hash ring." Each node's identifier is hashed to a point on the ring; each object's key is hashed similarly and assigned to the first node clockwise from that point. The key property is *minimal remapping*: when a node joins or leaves, only the keys between the arriving/departing node and its clockwise predecessor must move — typically a K/N fraction of all keys (K = key count, N = node count), far better than the near-total reshuffling required by naive mod-N partitioning.

Because a small number of physical nodes can produce uneven segments, production systems add **virtual nodes**: each physical node claims many pseudo-random positions on the ring, smoothing the distribution so that each node receives approximately equal share, with variance shrinking as the virtual-node count grows.

Consistent hashing underpins distributed caches (Memcached via Ketama, Dynamo-style key-value stores such as Amazon DynamoDB and Apache Cassandra) and content-addressable CDNs. It enables elastic scaling — nodes can be added or removed without global rehashing — while preserving locality and offering O(1) lookup via sorted-ring traversal or binary search.
