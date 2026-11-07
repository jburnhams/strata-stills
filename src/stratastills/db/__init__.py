"""The shared database (schema `photos` in the `omarchy` Postgres, see jburnhams/database docs/ADOPTING.md): everything about the pictures that is not the pictures themselves. Needs the optional packages in requirements-db.txt (omarchy-db, pgvector, psycopg); nothing else in stratastills imports this package yet.

The `vector` extension lives in schema `public` of `omarchy`, but omarchy_db migrates with search_path set to the app's schema alone, so `types.Vector` writes the type schema-qualified (`public.vector`) in DDL; at run time the URL should still say `search_path=photos,public`."""
