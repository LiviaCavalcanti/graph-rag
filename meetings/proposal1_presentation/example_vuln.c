struct sock;
struct socket { struct sock *sk; };
struct chan;

struct sock *sk_alloc(void);
void sk_free(struct sock *sk);
struct chan *chan_create(void);
void sock_init(struct socket *sock, struct sock *sk);

struct sock *alloc_socket(struct socket *sock) {
    struct sock *sk;
    struct chan *ch;

    sk = sk_alloc();
    if (!sk) return NULL;

    sock->sk = sk;

    ch = chan_create();
    if (!ch) {
        sk_free(sk);
        return NULL;
    }
    return sk;
}
