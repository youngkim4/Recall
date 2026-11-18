"""The functions the server calls: the archive behind the shapes of the routes.

Each module answers one route, or one concern that several routes share:

    recall_services.archive_state   where the archive is, one read handle, the two ways to fill it
    recall_services.names           the one place for names
    recall_services.search          /api/search
    recall_services.day             /api/day
    recall_services.conversations   the rows of /api/contacts and the archive totals
    recall_services.frame           the archive as the frame that the figures on pandas read
    recall_services.chat            /api/ask and /api/ask/stream
    recall_services.memory          the two optional ports of chat: search by meaning, the digest
    recall_services.meaning_port    the search by meaning of chat over the index of one archive
    recall_services.meaning         /api/meaning and the job that builds the index of meanings
    recall_services.digest_job      the job that reads one conversation for chat outside a question
    recall_services.reports         the estimate, a report run, the moments of a report
    recall_services.models          the models, their limits and their prices

No module here imports `ui_server`. Importing this package imports none of
the modules above, so it needs no model SDK and no pandas.
"""
