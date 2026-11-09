-- Chembed evaluation: four application tables (plan.md §3).
-- Access model: users authenticate with Supabase Auth; application tables are
-- exclusively reachable through service_role. Row-level security is enabled so
-- that no table is silently exposed, and every direct privilege for
-- PUBLIC/anon/authenticated is revoked before service_role is granted access.

-- ---------------------------------------------------------------- passages

create table public.passages (
    id                    bigint generated always as identity primary key,
    original_passage_id   text not null unique,
    text                  text not null,
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now()
);

comment on table public.passages is
    'Evaluation corpus passages. original_passage_id keeps the source dataset ID; text stored once per passage.';

-- ---------------------------------------------------------------- searches

create table public.searches (
    id               uuid not null default gen_random_uuid() primary key,
    user_id          uuid not null references auth.users (id) on delete restrict,
    query            text not null,
    information_need text,
    created_at       timestamptz not null default now(),
    status           text not null default 'pending'
                     check (status in ('pending', 'ready', 'failed')),
    is_practice      boolean not null default false,
    settings         jsonb not null
);

comment on column public.searches.status is
    'pending = created before model results stored; ready = results saved; failed = a model search failed. Query completeness is derived from ratings, not status.';
comment on column public.searches.settings is
    'Provenance snapshot: model revisions, corpus version/hash, Pinecone namespace, top-k, encoding/scoring settings.';

-- ---------------------------------------------------------------- results

create table public.results (
    id                uuid not null default gen_random_uuid() primary key,
    search_id         uuid not null references public.searches (id) on delete cascade,
    passage_id        bigint not null references public.passages (id),
    base_rank         integer check (base_rank between 1 and 5),
    base_score        double precision,
    progressive_rank  integer check (progressive_rank between 1 and 5),
    progressive_score double precision,
    check (base_rank is not null or progressive_rank is not null),
    check ((base_rank is null) = (base_score is null)),
    check ((progressive_rank is null) = (progressive_score is null)),
    display_position integer not null check (display_position between 1 and 10),
    constraint results_search_passage_unique unique (search_id, passage_id),
    -- Per search, each model appears with a given rank at most once.
    constraint results_search_model_base_rank_unique
        unique (search_id, base_rank),
    constraint results_search_model_prog_rank_unique
        unique (search_id, progressive_rank),
    -- Per search, each display position is used once.
    constraint results_search_display_pos_unique
        unique (search_id, display_position)
);

-- ---------------------------------------------------------------- ratings

create table public.ratings (
    id         uuid not null default gen_random_uuid() primary key,
    result_id  uuid not null unique references public.results (id) on delete cascade,
    relevant   boolean,
    note       text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

comment on table public.ratings is
    'One rating per displayed result, applied to both models when both retrieved the passage. relevant: true = relevant, false = not relevant, NULL = unanswered/uncertain (with optional note); a NULL keeps the search incomplete.';

-- ---------------------------------------------------------------- indexes

create index searches_user_id_idx  on public.searches (user_id);
create index results_passage_id_idx on public.results (passage_id);

-- ---------------------------------------------------------------- locking down

alter table public.passages  enable row level security;
alter table public.searches  enable row level security;
alter table public.results   enable row level security;
alter table public.ratings   enable row level security;

revoke all on public.passages from public, anon, authenticated;
revoke all on public.searches from public, anon, authenticated;
revoke all on public.results  from public, anon, authenticated;
revoke all on public.ratings  from public, anon, authenticated;

grant select, insert, update, delete on public.passages, public.searches, public.results, public.ratings to service_role;

revoke all on sequence public.passages_id_seq from public, anon, authenticated;
grant usage, select on sequence public.passages_id_seq to service_role;
